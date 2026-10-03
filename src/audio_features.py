"""Per-second audio excitement features: RMS, spectral flux, crowd-band energy, robust z-scores."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .utils import VideoReadError, get_logger

log = get_logger(__name__)

N_FFT = 1024          # 64 ms at 16 kHz
HOP = 400             # 25 ms at 16 kHz -> 40 frames per second
BLOCK_SECONDS = 60    # stream the WAV in 60 s blocks (frames are aligned to whole seconds)
EPS = 1e-8
Z_CLIP = (-3.0, 6.0)


def _frame_features(wav_path: Path, sr: int, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the WAV and return per-hop (25 ms) mean-square, spectral flux and crowd-band amplitude."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    frames_per_sec = sr // HOP
    block_frames = BLOCK_SECONDS * frames_per_sec

    msq, flux, crowd = [], [], []
    prev_last: np.ndarray | None = None
    for y in librosa.stream(str(wav_path), block_length=block_frames, frame_length=N_FFT,
                            hop_length=HOP, mono=True, sr=None):
        n = (len(y) - N_FFT) // HOP + 1
        if n < 1:
            continue
        # center=False: frame j covers samples [j*HOP, j*HOP+N_FFT), so blocks line up with whole seconds
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=False)).astype(np.float32)
        msq.append(np.mean(np.square(y[: n * HOP], dtype=np.float64).reshape(n, HOP), axis=1))
        d = np.diff(S, axis=1, prepend=(prev_last if prev_last is not None else S[:, :1]))
        flux.append(np.maximum(d, 0).sum(axis=0))
        crowd.append(np.sqrt(np.mean(np.square(S[band_mask], dtype=np.float64), axis=0)))
        prev_last = S[:, -1:]
    if not msq:
        raise VideoReadError(f"Audio file {wav_path} is empty or too short.")
    return np.concatenate(msq), np.concatenate(flux), np.concatenate(crowd)


def _per_second(values: np.ndarray, n_seconds: int, frames_per_sec: int, reduce: str) -> np.ndarray:
    """Aggregate hop-level values into n_seconds bins, zero-filling any missing seconds."""
    out = np.zeros(n_seconds)
    usable = min(len(values) // frames_per_sec, n_seconds)
    if usable > 0:
        block = values[: usable * frames_per_sec].reshape(usable, frames_per_sec)
        out[:usable] = np.sqrt(block.mean(axis=1)) if reduce == "rms" else block.mean(axis=1)
    return out


def robust_z(x: pd.Series, window_s: int) -> pd.Series:
    """Robust z = (x - rolling_median) / (1.4826 * rolling_MAD + eps), centered window."""
    window = max(3, int(window_s))
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    return (x - med) / (1.4826 * mad + EPS)


def compute_audio_features(wav_path: str | Path, cfg: dict) -> pd.DataFrame:
    """Return per-second DataFrame: second, rms, flux, crowd, rms_z, flux_z, crowd_z, audio_z."""
    wav_path = Path(wav_path)
    acfg = cfg["audio"]
    if not wav_path.is_file():
        raise VideoReadError(f"Audio file not found: {wav_path}")
    info = sf.info(str(wav_path))
    sr = info.samplerate
    if sr % HOP != 0:
        raise VideoReadError(f"Unsupported sample rate {sr}; expected the 16 kHz WAV from audio_extractor.")
    n_seconds = int(info.frames // sr)
    if n_seconds < 1:
        raise VideoReadError(f"Audio file {wav_path} is shorter than one second.")
    log.info("Computing audio features for %d s of audio", n_seconds)

    fps = sr // HOP
    msq, flux, crowd = _frame_features(wav_path, sr, tuple(acfg["crowd_band_hz"]))
    df = pd.DataFrame({
        "second": np.arange(n_seconds),
        "rms": _per_second(msq, n_seconds, fps, "rms"),
        "flux": _per_second(flux, n_seconds, fps, "mean"),
        "crowd": _per_second(crowd, n_seconds, fps, "mean"),
    })
    for col in ("rms", "flux", "crowd"):
        df[f"{col}_z"] = robust_z(df[col], acfg["baseline_window_s"])
    df["audio_z"] = df[["rms_z", "flux_z", "crowd_z"]].mean(axis=1).clip(*Z_CLIP)
    return df
