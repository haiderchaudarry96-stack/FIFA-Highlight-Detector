"""Phase 2 check: transcribe commentary + keyword scores, print the 10 top keyword seconds.

Usage: .venv\\Scripts\\python.exe scripts/check_commentary.py samples/test_match.mp4 [--model tiny] [--no-cache]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_extractor import extract_audio  # noqa: E402
from src.commentary import get_snippet, score_keywords, transcribe  # noqa: E402
from src.utils import (FFmpegNotFoundError, NoAudioTrackError, VideoReadError, check_ffmpeg,  # noqa: E402
                       fmt_time, get_logger, get_output_dirs, get_video_duration, is_cache_valid,
                       load_config)

log = get_logger("check_commentary")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", help="whisper model size (default: config whisper.model)")
    ap.add_argument("--no-cache", action="store_true", help="recompute even if cache files exist")
    args = ap.parse_args()

    try:
        check_ffmpeg()
        cfg = load_config(args.config)
        wcfg = cfg["whisper"]
        model = args.model or wcfg["model"]
        video = args.video
        n_seconds = int(get_video_duration(video))
        dirs = get_output_dirs(video)
        wav = dirs["cache"] / "audio.wav"
        t0 = time.time()

        if not args.no_cache and is_cache_valid(wav, video):
            log.info("Using cached audio: %s", wav)
        else:
            extract_audio(video, wav, cfg["audio"]["sample_rate"])

        cache = dirs["cache"] / f"transcript_{model}.json"
        if args.no_cache and cache.exists():
            cache.unlink()
        words = transcribe(wav, model, wcfg.get("backend", "auto"), cache, wcfg.get("language"))
    except (FFmpegNotFoundError, NoAudioTrackError, VideoReadError, FileNotFoundError, ImportError) as e:
        log.error("%s", e)
        return 1

    df = score_keywords(words, cfg["keywords"], cfg["keyword_options"], n_seconds)
    kw_csv = dirs["cache"] / "keywords.csv"
    df.to_csv(kw_csv, index=False)
    log.info("Saved %s (done in %.1fs)", kw_csv, time.time() - t0)

    n_hits = sum(len(h.split("; ")) for h in df["keyword_hits"] if h)
    print(f"\nTranscript: {len(words)} words, last word at "
          f"{fmt_time(words[-1]['end']) if words else '--:--'} (video {fmt_time(n_seconds)})")
    print(f"Total keyword hits: {n_hits}")

    spread = int(cfg["keyword_options"].get("spread_s", 0))
    # One second per moment: a spread hit covers 2*spread_s+1 adjacent seconds, so skip
    # seconds within that distance of an already chosen (higher) one. Ties -> earliest.
    gap = 2 * spread + 1
    chosen: list[int] = []
    for sec in df[df["keyword_score"] > 0].sort_values(["keyword_score", "second"],
                                                       ascending=[False, True])["second"]:
        if all(abs(sec - c) >= gap for c in chosen):
            chosen.append(int(sec))
        if len(chosen) == 10:
            break
    top = df.loc[chosen]
    print(f"\nTop 10 keyword moments ({model}, at least {gap}s apart):")
    for rank, row in enumerate(top.itertuples(), 1):
        near = df.loc[max(0, row.second - spread): row.second + spread, "keyword_hits"]
        hits = "; ".join(h for h in near if h)
        snippet = get_snippet(words, row.second - 5, row.second + 5)
        print(f"{rank:2d}. {fmt_time(row.second)}  score={row.keyword_score:4.2f}  hits[+/-{spread}s]: {hits}")
        print(f"    \"{snippet}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
