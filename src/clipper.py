"""Cut highlight clips, thumbnails and the stitched reel with ffmpeg (subprocess, no moviepy)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import VideoReadError, check_ffmpeg, get_logger, get_video_duration

log = get_logger(__name__)

MIN_CLIP_BYTES = 10 * 1024   # a stream-copied clip smaller than this is treated as broken
SHORT_TOLERANCE_S = 2.0      # copied clip may be at most this much shorter than asked
LONG_TOLERANCE_S = 10.0      # ... or this much longer (stream copy starts at the keyframe before `start`)


def build_windows(peaks: list[int], pre_s: float, post_s: float,
                  duration: float) -> list[tuple[float, float, list[int]]]:
    """[(start, end, peaks_in_window)] sorted by start: each peak gets [p - pre_s, p + post_s]
    clamped to [0, duration]; overlapping (or touching) windows are merged."""
    windows: list[tuple[float, float, list[int]]] = []
    for p in sorted(peaks):
        start = max(0.0, float(p) - pre_s)
        end = min(float(duration), float(p) + post_s)
        if end <= start:
            continue
        if windows and start <= windows[-1][1]:
            s, e, ps = windows[-1]
            windows[-1] = (s, max(e, end), ps + [p])
        else:
            windows.append((start, end, [p]))
    return windows


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """Run ffmpeg quietly with an argument list (overwrites outputs)."""
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def _probe_duration(path: Path) -> float | None:
    """Duration of a produced file, or None if it is missing/unreadable."""
    if not path.is_file():
        return None
    try:
        return get_video_duration(path)
    except VideoReadError:
        return None


def _clip_ok(path: Path, expected_s: float) -> bool:
    """True if the clip exists, is not tiny and its duration is close to what was asked."""
    if not path.is_file() or path.stat().st_size < MIN_CLIP_BYTES:
        return False
    d = _probe_duration(path)
    return d is not None and expected_s - SHORT_TOLERANCE_S <= d <= expected_s + LONG_TOLERANCE_S


def cut_clip(video_path: str | Path, start: float, end: float, out_path: str | Path) -> Path:
    """Cut [start, end] into out_path. Tries stream copy first (fast, keyframe-aligned start);
    re-encodes with libx264 (veryfast) + aac if the copy is missing, tiny or the wrong length."""
    check_ffmpeg()
    video_path, out_path = Path(video_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    length = max(0.0, end - start)
    tmp = out_path.with_name(out_path.stem + ".part.mp4")
    seek = ["-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{length:.3f}"]  # -ss before -i = fast seek

    proc = _run_ffmpeg([*seek, "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(tmp)])
    if proc.returncode == 0 and _clip_ok(tmp, length):
        tmp.replace(out_path)
        return out_path

    log.info("Stream copy of %.1f-%.1f s not usable, re-encoding %s", start, end, out_path.name)
    proc = _run_ffmpeg([*seek, "-map", "0:v:0", "-map", "0:a:0?",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)])
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise VideoReadError(f"ffmpeg could not cut {start:.1f}-{end:.1f} s from {video_path.name}: "
                             f"{proc.stderr.strip()[-500:] or 'unknown error'}")
    tmp.replace(out_path)
    return out_path


def save_thumbnail(video_path: str | Path, t: float, out_jpg: str | Path, width: int = 480) -> Path:
    """Save one frame at time t (seconds) as a JPEG, `width` px wide."""
    check_ffmpeg()
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    proc = _run_ffmpeg(["-ss", f"{max(0.0, t):.3f}", "-i", str(video_path), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", "-q:v", "3", str(out_jpg)])
    if proc.returncode != 0 or not out_jpg.is_file():
        raise VideoReadError(f"ffmpeg could not save a thumbnail at {t:.1f} s: "
                             f"{proc.stderr.strip()[-500:] or 'unknown error'}")
    return out_jpg


def build_reel(clip_paths: list[str | Path], out_path: str | Path) -> Path:
    """Concatenate clips (in the given, chronological order) with the concat demuxer.
    Tries stream copy; re-encodes (libx264 veryfast + aac) if that fails or the length is off."""
    check_ffmpeg()
    clips = [Path(p) for p in clip_paths]
    if not clips:
        raise ValueError("build_reel needs at least one clip")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_file = out_path.with_name(out_path.stem + "_concat.txt")
    # concat demuxer syntax: file '<path>'; forward slashes, single quotes escaped as '\''
    lines = ["file '" + p.resolve().as_posix().replace("'", "'\\''") + "'" for p in clips]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    expected = sum(d for p in clips if (d := _probe_duration(p)) is not None)
    tmp = out_path.with_name(out_path.stem + ".part.mp4")
    concat = ["-f", "concat", "-safe", "0", "-i", str(list_file)]

    try:
        proc = _run_ffmpeg([*concat, "-c", "copy", "-movflags", "+faststart", str(tmp)])
        d = _probe_duration(tmp) if proc.returncode == 0 else None
        if d is not None and abs(d - expected) <= max(2.0, 0.05 * expected):
            tmp.replace(out_path)
            return out_path

        log.info("Stream-copy concat not usable (got %s, expected %.1f s), re-encoding the reel",
                 f"{d:.1f} s" if d is not None else "no file", expected)
        proc = _run_ffmpeg([*concat, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                            "-movflags", "+faststart", str(tmp)])
        if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise VideoReadError(f"ffmpeg could not build the reel: "
                                 f"{proc.stderr.strip()[-500:] or 'unknown error'}")
        tmp.replace(out_path)
        return out_path
    finally:
        list_file.unlink(missing_ok=True)
