"""Shared helpers: config, ffmpeg check, paths, cache rule, time formatting, logging, robust z-score."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPS = 1e-8


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg/ffprobe is not installed or not on PATH."""


class NoAudioTrackError(RuntimeError):
    """The video file has no audio stream."""


class VideoReadError(RuntimeError):
    """The video file is missing, corrupt or unreadable."""


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load the YAML config. Relative paths are tried as given, then against the project root."""
    p = Path(path)
    if not p.is_file() and not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_ffmpeg() -> None:
    """Raise FFmpegNotFoundError (with install hints) if ffmpeg or ffprobe is missing."""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise FFmpegNotFoundError(
            f"{', '.join(missing)} not found on PATH. Install ffmpeg:\n"
            "  Windows: winget install ffmpeg   (then reopen the terminal)\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg"
        )


def _ffprobe(args: list[str]) -> str:
    """Run ffprobe with an argument list and return stdout."""
    check_ffmpeg()
    proc = subprocess.run(
        ["ffprobe", "-v", "error", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise VideoReadError(f"ffprobe failed: {proc.stderr.strip() or 'unknown error'}")
    return proc.stdout.strip()


def get_video_duration(video_path: str | Path) -> float:
    """Return the container duration in seconds (ffprobe)."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise VideoReadError(f"Video file not found: {video_path}")
    out = _ffprobe(["-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video_path)])
    try:
        return float(out)
    except ValueError:
        raise VideoReadError(f"Could not read duration of {video_path} (corrupt or unsupported file?)")


def get_output_dirs(video_path: str | Path, root: str | Path = "outputs") -> dict[str, Path]:
    """Create and return outputs/<video_stem>/{cache,clips,thumbs}. Keys: root, cache, clips, thumbs."""
    root = Path(root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    base = root / Path(video_path).stem
    dirs = {"root": base, "cache": base / "cache", "clips": base / "clips", "thumbs": base / "thumbs"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def is_cache_valid(cache_file: str | Path, source_file: str | Path) -> bool:
    """True if cache_file exists, is non-empty and is newer than source_file."""
    cache_file, source_file = Path(cache_file), Path(source_file)
    if not cache_file.is_file() or cache_file.stat().st_size == 0 or not source_file.is_file():
        return False
    return cache_file.stat().st_mtime >= source_file.stat().st_mtime


def fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss (minutes keep growing past 59, e.g. 95:03)."""
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with one console handler (configured once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def robust_z(x: pd.Series, window_s: int) -> pd.Series:
    """Robust z = (x - rolling_median) / (1.4826 * rolling_MAD + eps), centered window."""
    window = max(3, int(window_s))
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    return (x - med) / (1.4826 * mad + EPS)
