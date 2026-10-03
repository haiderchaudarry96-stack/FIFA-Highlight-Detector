"""Command line entry: find highlights in a match video and cut clips + a reel.

Usage: .venv\\Scripts\\python.exe cli.py --video samples/test_match.mp4 [--top 5] [--model tiny]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.utils import FFmpegNotFoundError, NoAudioTrackError, VideoReadError, load_config


def print_table(highlights: list[dict]) -> None:
    """Print rank, time, type, score and reason as an aligned table (reason wraps to the terminal)."""
    if not highlights:
        print("No highlights found.")
        return
    type_w = max(len("label"), *(len(h["event_type"]) for h in highlights))
    head = f"{'rank':>4}  {'time':>5}  {'label':<{type_w}}  {'score':>5}  "
    reason_w = max(30, shutil.get_terminal_size((120, 20)).columns - len(head) - 1)
    print(head + "reason")
    print("-" * min(len(head) + reason_w, 160))
    for h in highlights:
        reason = h["reason"]
        lines = [reason[i:i + reason_w] for i in range(0, len(reason), reason_w)] or [""]
        print(f"{h['rank']:>4}  {h['timestamp']:>5}  {h['event_type']:<{type_w}}  {h['score']:5.3f}  {lines[0]}")
        for extra in lines[1:]:
            print(" " * len(head) + extra)


def main() -> int:
    ap = argparse.ArgumentParser(description="Find highlights in a football match video.")
    ap.add_argument("--video", required=True, type=Path, help="match video (mp4/mkv)")
    ap.add_argument("--top", type=int, help="number of highlights (default: scoring.top_n)")
    ap.add_argument("--model", choices=["tiny", "base", "small"], help="Whisper model (default: whisper.model)")
    ap.add_argument("--pre", type=float, help="seconds before each peak (default: clips.pre_s)")
    ap.add_argument("--post", type=float, help="seconds after each peak (default: clips.post_s)")
    ap.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    ap.add_argument("--no-cache", action="store_true", help="recompute everything, ignore cached features")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.top is not None:
        cfg["scoring"]["top_n"] = args.top
    if args.model:
        cfg["whisper"]["model"] = args.model
    cfg.setdefault("clips", {})
    if args.pre is not None:
        cfg["clips"]["pre_s"] = args.pre
    if args.post is not None:
        cfg["clips"]["post_s"] = args.post

    from src.pipeline import run_pipeline

    try:
        result = run_pipeline(args.video, cfg, use_cache=not args.no_cache)
    except (FFmpegNotFoundError, NoAudioTrackError, VideoReadError, FileNotFoundError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\nHighlights of {args.video.name} (ranked by score):\n")
    print_table(result.highlights)
    print(f"\nOutput folder: {result.out_dir}")
    if result.reel_path:
        print(f"Reel:          {result.reel_path}")
    print(f"Highlights:    {result.json_path.name}, {result.csv_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
