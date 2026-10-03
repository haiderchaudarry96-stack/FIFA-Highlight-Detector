"""Extract a mono 16 kHz WAV from a video with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import NoAudioTrackError, VideoReadError, check_ffmpeg, get_logger

log = get_logger(__name__)


def has_audio_track(video_path: str | Path) -> bool:
    """True if ffprobe finds at least one audio stream."""
    check_ffmpeg()
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise VideoReadError(f"ffprobe could not read {video_path}: {proc.stderr.strip()}")
    return bool(proc.stdout.strip())


def extract_audio(video_path: str | Path, out_wav: str | Path, sample_rate: int = 16000) -> Path:
    """Write mono `sample_rate` Hz 16-bit WAV of the first audio track to out_wav.

    Raises NoAudioTrackError if the video has no audio, VideoReadError if ffmpeg fails.
    """
    video_path, out_wav = Path(video_path), Path(out_wav)
    if not video_path.is_file():
        raise VideoReadError(f"Video file not found: {video_path}")
    if not has_audio_track(video_path):
        raise NoAudioTrackError(f"{video_path.name} has no audio track; cannot detect audio highlights.")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_name(out_wav.stem + ".part.wav")  # avoid leaving a half-written cache file
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-map", "0:a:0",
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)]
    log.info("Extracting audio -> %s", out_wav)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise VideoReadError(f"ffmpeg audio extraction failed: {proc.stderr.strip() or 'empty output'}")
    tmp.replace(out_wav)
    return out_wav
