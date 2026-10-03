"""Phase 3 check: video features (motion, cuts, replays), speed, memory, top 10 seconds by video_z.

Usage: .venv\\Scripts\\python.exe scripts/check_video.py samples/test_match.mp4 [--no-cache]
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.utils import (VideoReadError, fmt_time, get_logger, get_output_dirs,  # noqa: E402
                       get_video_duration, is_cache_valid, load_config)
from src.video_features import compute_video_features  # noqa: E402

log = get_logger("check_video")


def rss_mb() -> float | None:
    """Current resident memory (working set) of this process in MB; None if unavailable."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 2**20
        return None
    try:
        with open("/proc/self/statm") as f:
            import os
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return None


def replay_ranges(replay: pd.Series) -> list[tuple[int, int]]:
    """Contiguous runs of replay == 1 as (first_second, last_second)."""
    ranges, start = [], None
    for sec, flag in enumerate(replay.to_numpy()):
        if flag and start is None:
            start = sec
        elif not flag and start is not None:
            ranges.append((start, sec - 1))
            start = None
    if start is not None:
        ranges.append((start, len(replay) - 1))
    return ranges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-cache", action="store_true", help="recompute even if the cache file exists")
    args = ap.parse_args()

    mem: list[tuple[float, float]] = []   # (fraction, RSS MB) at each progress callback

    def progress(frac: float) -> None:
        m = rss_mb()
        if m is not None:
            mem.append((frac, m))
        log.info("video %3.0f%%  RSS %s MB", frac * 100, f"{m:.0f}" if m is not None else "?")

    try:
        cfg = load_config(args.config)
        video = args.video
        duration = get_video_duration(video)
        feat_csv = get_output_dirs(video)["cache"] / "video_features.csv"
        t0 = time.time()
        if not args.no_cache and is_cache_valid(feat_csv, video):
            log.info("Using cached video features: %s (use --no-cache to measure speed)", feat_csv)
            df = pd.read_csv(feat_csv)
        else:
            start_mem = rss_mb()
            df = compute_video_features(video, cfg, progress_cb=progress)
            df.to_csv(feat_csv, index=False)
            log.info("Saved %s", feat_csv)
            elapsed = time.time() - t0
            print(f"\nProcessed {fmt_time(duration)} of video in {elapsed:.1f}s "
                  f"-> {duration / elapsed:.1f} s of video per s of compute")
            if mem and start_mem is not None:
                rss = [m for _, m in mem]
                early = next((m for f, m in mem if f >= 0.1), rss[0])
                print(f"Memory (RSS): start {start_mem:.0f} MB, at 10% {early:.0f} MB, "
                      f"peak {max(rss):.0f} MB, end {rss[-1]:.0f} MB "
                      f"(growth 10%->end {rss[-1] - early:+.0f} MB)")
    except (VideoReadError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1

    vcfg = cfg["video"]
    print(f"\n{len(df)} s of features; mean motion {df['motion'].mean():.2f}, "
          f"{int(df['cuts'].sum())} cuts (threshold {vcfg['cut_threshold']}), "
          f"{int(df['replay'].sum())} s flagged replay")

    top = df.sort_values(["video_z", "second"], ascending=[False, True]).head(10)
    print(f"\nTop 10 seconds of {video.name} by video_z:")
    for rank, row in enumerate(top.itertuples(), 1):
        print(f"{rank:2d}. {fmt_time(row.second)}  video_z={row.video_z:5.2f}  motion_z={row.motion_z:5.2f}  "
              f"cuts={row.cuts}  replay={row.replay}")

    ranges = replay_ranges(df["replay"])
    print(f"\nReplay ranges ({len(ranges)}; >= {vcfg['replay_min_cuts']} cuts in a "
          f"{vcfg['replay_window_s']} s window):")
    for a, b in ranges:
        print(f"  {fmt_time(a)} - {fmt_time(b)}  ({b - a + 1} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
