"""Per-second visual features: motion, scene cuts, replay flag and a combined robust video_z."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .utils import FFmpegNotFoundError, VideoReadError, get_logger, get_video_duration, robust_z

log = get_logger(__name__)

HIST_BINS = [30, 32]            # hue x saturation bins for the scene-cut histogram
HIST_RANGES = [0, 180, 0, 256]
CUT_TERM_MAX = 3.0              # cut density at replay level counts like a z-score of 3
W_MOTION, W_CUTS = 0.6, 0.4
Z_CLIP = (-3.0, 6.0)
PROGRESS_STEP = 0.05


def _open(video_path: Path) -> tuple[cv2.VideoCapture, float]:
    """Open the video with OpenCV; raise VideoReadError if it can't be opened or has no fps."""
    if not video_path.is_file():
        raise VideoReadError(f"Video file not found: {video_path}")
    # Ask for hardware decoding (Intel iGPU via D3D11/QSV); OpenCV silently falls back to software.
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"OpenCV could not open {video_path} (corrupt file or unsupported codec, e.g. AV1?)")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise VideoReadError(f"Video {video_path} reports fps=0 (corrupt or unsupported file).")
    return cap, float(fps)


def _n_seconds(video_path: Path, cap: cv2.VideoCapture, fps: float) -> int:
    """N = floor(container duration) via ffprobe, like the other per-second tables; frame count as fallback."""
    try:
        return int(get_video_duration(video_path))
    except (FFmpegNotFoundError, VideoReadError) as e:
        log.warning("ffprobe duration failed (%s); using OpenCV frame count", e)
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)


def _hs_hist(small_bgr: np.ndarray) -> np.ndarray:
    """Hue/saturation histogram of a (resized) BGR frame."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    return cv2.calcHist([hsv], [0, 1], None, HIST_BINS, HIST_RANGES)


def compute_video_features(video_path: str | Path, cfg: dict,
                           progress_cb: Callable[[float], None] | None = None) -> pd.DataFrame:
    """Return per-second DataFrame: second, motion, cuts, replay, motion_z, video_z.

    Streams the video: every frame is `grab()`bed (cheap, no colour conversion) and only frames at
    `sample_fps` are `retrieve()`d. Only the previous sampled frame is kept, so memory stays flat.
    """
    video_path = Path(video_path)
    vcfg = cfg["video"]
    sample_fps = float(vcfg["sample_fps"])
    resize_w = int(vcfg["resize_width"])
    cut_thr = float(vcfg["cut_threshold"])
    window_s = int(vcfg.get("baseline_window_s", cfg["audio"]["baseline_window_s"]))

    cap, fps = _open(video_path)
    try:
        n_seconds = _n_seconds(video_path, cap, fps)
        if n_seconds < 1:
            raise VideoReadError(f"Video {video_path} is shorter than one second.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(n_seconds * fps)
        log.info("Computing video features for %d s of video (%.2f fps, sampling %.1f fps)",
                 n_seconds, fps, sample_fps)

        motion_sum = np.zeros(n_seconds)
        motion_cnt = np.zeros(n_seconds, dtype=np.int32)
        cuts = np.zeros(n_seconds, dtype=np.int32)
        seen = np.zeros(n_seconds, dtype=bool)    # second had at least one sampled frame

        prev_gray: np.ndarray | None = None
        prev_hist: np.ndarray | None = None
        size: tuple[int, int] | None = None
        next_t = 0.0
        next_report = PROGRESS_STEP
        idx = 0
        t0 = time.time()

        while cap.grab():
            t = idx / fps
            idx += 1
            if progress_cb and idx / total_frames >= next_report:
                progress_cb(min(idx / total_frames, 1.0))
                next_report += PROGRESS_STEP
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / sample_fps
            sec = int(t)
            if sec >= n_seconds:
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                prev_gray = prev_hist = None   # don't diff across an unreadable frame
                continue
            if size is None:
                h, w = frame.shape[:2]
                size = (resize_w, max(1, round(h * resize_w / w)))
            small = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)  # 15x faster than INTER_AREA
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = _hs_hist(small)
            seen[sec] = True

            if prev_gray is not None:
                is_cut = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < cut_thr
                if is_cut:
                    cuts[sec] += 1
                else:
                    # a cut's frame difference is a shot change, not motion -> leave it out
                    motion_sum[sec] += float(cv2.absdiff(gray, prev_gray).mean())
                    motion_cnt[sec] += 1
            prev_gray, prev_hist = gray, hist

        last_sec = int(idx / fps)
        if last_sec < n_seconds - 2:
            log.warning("Video stream ended at %d s of %d s (truncated or corrupt file?); rest filled with 0",
                        last_sec, n_seconds)
    finally:
        cap.release()

    if progress_cb:
        progress_cb(1.0)
    elapsed = time.time() - t0
    log.info("Decoded %d frames in %.1fs (%.1fx realtime)", idx, elapsed, (idx / fps) / max(elapsed, 1e-6))

    # Seconds whose only frame pairs were cuts get interpolated motion; seconds never reached stay 0.
    motion = pd.Series(np.where(motion_cnt > 0, motion_sum / np.maximum(motion_cnt, 1), np.nan))
    motion[~seen] = 0.0
    motion = motion.interpolate(limit_area="inside").fillna(0.0)

    df = pd.DataFrame({"second": np.arange(n_seconds), "motion": motion.to_numpy(), "cuts": cuts})
    min_cuts = int(vcfg["replay_min_cuts"])
    window_cuts = df["cuts"].rolling(int(vcfg["replay_window_s"]), center=True, min_periods=1).sum()
    df["replay"] = (window_cuts >= min_cuts).astype(int)
    df["motion_z"] = robust_z(df["motion"], window_s)
    # cut density in the replay window scaled to 0..CUT_TERM_MAX (replay level = max)
    cut_term = CUT_TERM_MAX * (window_cuts / min_cuts).clip(0, 1)
    df["video_z"] = (W_MOTION * df["motion_z"] + W_CUTS * cut_term).clip(*Z_CLIP)
    return df
