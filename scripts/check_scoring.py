"""Phase 4 check: fuse cached features, pick the top-N highlights, classify them, compare with the truth file.

Usage: .venv\\Scripts\\python.exe scripts/check_scoring.py samples/test_match.mp4 [--truth samples/truth_test.csv]
Features are loaded from outputs/<stem>/cache/ and only computed if a cache file is missing or stale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.classifier import classify  # noqa: E402
from src.scorer import combine_signals, find_highlights  # noqa: E402
from src.utils import (FFmpegNotFoundError, NoAudioTrackError, VideoReadError, fmt_time,  # noqa: E402
                       get_logger, get_output_dirs, get_video_duration, is_cache_valid, load_config)

log = get_logger("check_scoring")
TOLERANCE_S = 20


def load_or_compute(video: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audio, keyword and video tables: from the cache if valid, else computed and cached."""
    cache = get_output_dirs(video)["cache"]
    wav = cache / "audio.wav"
    out = {}
    for name, fname in (("audio", "audio_features.csv"), ("keywords", "keywords.csv"),
                        ("video", "video_features.csv")):
        path = cache / fname
        if is_cache_valid(path, video):
            log.info("Using cached %s", path)
            out[name] = pd.read_csv(path, keep_default_na=False) if name == "keywords" else pd.read_csv(path)
            continue
        log.info("No valid cache for %s, computing ...", fname)
        if name in ("audio", "keywords") and not is_cache_valid(wav, video):
            from src.audio_extractor import extract_audio
            extract_audio(video, wav, cfg["audio"]["sample_rate"])
        if name == "audio":
            from src.audio_features import compute_audio_features
            df = compute_audio_features(wav, cfg)
        elif name == "keywords":
            from src.commentary import score_keywords, transcribe
            w = cfg["whisper"]
            words = transcribe(wav, w["model"], w.get("backend", "auto"),
                               cache / f"transcript_{w['model']}.json", w.get("language"))
            df = score_keywords(words, cfg["keywords"], cfg["keyword_options"], int(get_video_duration(video)))
        else:
            from src.video_features import compute_video_features
            df = compute_video_features(video, cfg)
        df.to_csv(path, index=False)
        out[name] = df
    return out["audio"], out["keywords"], out["video"]


def load_truth(path: Path) -> list[tuple[int, str]]:
    """(second, event) rows from a `time,event` csv with mm:ss times."""
    df = pd.read_csv(path, skipinitialspace=True, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    rows = []
    for t, ev in zip(df["time"], df["event"]):
        parts = [int(p) for p in str(t).strip().split(":")]
        sec = 0
        for p in parts:
            sec = sec * 60 + p
        rows.append((sec, str(ev).strip()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--truth", type=Path, default=Path("samples/truth_test.csv"))
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
        audio_df, kw_df, video_df = load_or_compute(args.video, cfg)
    except (FFmpegNotFoundError, NoAudioTrackError, VideoReadError, FileNotFoundError, ImportError) as e:
        log.error("%s", e)
        return 1

    sc = cfg["scoring"]
    df = combine_signals(audio_df, kw_df, video_df, sc["weights"], sc["smooth_window_s"], sc)
    prominence = float(sc.get("min_prominence", 0.02))
    peaks = find_highlights(df["score"], sc["top_n"], sc["min_peak_distance_s"], prominence)
    all_peaks = find_highlights(df["score"], len(df), sc["min_peak_distance_s"], prominence)

    print(f"\nTop {len(peaks)} highlights of {args.video.name} ({len(all_peaks)} peaks in total):")
    print(f"{'rank':>4}  {'time':>5}  {'score':>5}  {'aud_n':>5}  {'kw_g':>5}  {'vid_n':>5}  "
          f"{'label':<18}  reason")
    for rank, p in enumerate(peaks, 1):
        r = df.loc[p]
        label, reason = classify(p, df, cfg)
        print(f"{rank:4d}  {fmt_time(p):>5}  {r.score:5.3f}  {r.audio_n:5.2f}  {r.keyword_gated:5.2f}  "
              f"{r.video_n:5.2f}  {label:<18}  {reason}")

    truth_path = args.truth if args.truth.is_absolute() else Path(__file__).resolve().parent.parent / args.truth
    if not truth_path.is_file():
        print(f"\n(no truth file at {args.truth}, skipping the goal check)")
        return 0
    goals = [s for s, ev in load_truth(truth_path) if ev.lower() == "goal"]
    print(f"\nTruth check ({truth_path.name}, +/-{TOLERANCE_S} s):")
    found = 0
    for g in goals:
        near = [(abs(p - g), rank, p) for rank, p in enumerate(peaks, 1)]
        hit = min((x for x in near if x[0] <= TOLERANCE_S), default=None)
        any_near = min(((abs(p - g), rank, p) for rank, p in enumerate(all_peaks, 1)), default=None)
        r = df.loc[min(g, len(df) - 1)]
        diag = (f"signals at goal: score={r.score:.3f} audio_z={r.audio_z:.1f} kw={r.keyword_score:.2f} "
                f"kw_gated={r.keyword_gated:.2f} replay_ahead={r.replay_ahead:.0f} video_z={r.video_z:.1f}")
        if hit:
            found += 1
            _, rank, p = hit
            label, _ = classify(p, df, cfg)
            print(f"  goal {fmt_time(g)}: FOUND  highlight #{rank} at {fmt_time(p)} "
                  f"(offset {p - g:+d} s, label {label})")
        else:
            msg = "MISSED"
            if any_near:
                d, rank, p = any_near
                msg += f"  nearest peak {fmt_time(p)} (offset {p - g:+d} s) is rank {rank} of {len(all_peaks)}"
            print(f"  goal {fmt_time(g)}: {msg}")
        print(f"      {diag}")
    print(f"  -> {found}/{len(goals)} goals in the top {len(peaks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
