"""Phase 1 check: extract audio + audio features for a video, print the 10 loudest seconds.

Usage: .venv\\Scripts\\python.exe scripts/check_audio.py samples/test_match.mp4 [--no-cache]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.audio_extractor import extract_audio  # noqa: E402
from src.audio_features import compute_audio_features  # noqa: E402
from src.utils import (NoAudioTrackError, FFmpegNotFoundError, VideoReadError, check_ffmpeg,  # noqa: E402
                       fmt_time, get_logger, get_output_dirs, get_video_duration, is_cache_valid,
                       load_config)

log = get_logger("check_audio")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-cache", action="store_true", help="recompute even if cache files exist")
    args = ap.parse_args()

    try:
        check_ffmpeg()
        cfg = load_config(args.config)
        video = args.video
        duration = get_video_duration(video)
        dirs = get_output_dirs(video)
        wav, feat_csv = dirs["cache"] / "audio.wav", dirs["cache"] / "audio_features.csv"
        t0 = time.time()

        if not args.no_cache and is_cache_valid(wav, video):
            log.info("Using cached audio: %s", wav)
        else:
            extract_audio(video, wav, cfg["audio"]["sample_rate"])

        if not args.no_cache and is_cache_valid(feat_csv, video):
            log.info("Using cached audio features: %s", feat_csv)
            df = pd.read_csv(feat_csv)
        else:
            df = compute_audio_features(wav, cfg)
            df.to_csv(feat_csv, index=False)
            log.info("Saved %s", feat_csv)
    except (FFmpegNotFoundError, NoAudioTrackError, VideoReadError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1

    log.info("Video %s, %d s of features, done in %.1fs", fmt_time(duration), len(df), time.time() - t0)
    top = df.nlargest(10, "audio_z")
    print(f"\n10 loudest seconds of {video.name} (by audio_z):")
    for rank, row in enumerate(top.itertuples(), 1):
        print(f"{rank:2d}. {fmt_time(row.second)}  audio_z={row.audio_z:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
