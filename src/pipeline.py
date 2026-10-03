"""End-to-end run: features -> fused score -> top highlights -> clips, thumbnails, reel, json/csv."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .utils import check_ffmpeg, fmt_time, get_logger, get_output_dirs, get_video_duration, is_cache_valid

log = get_logger(__name__)

ProgressCb = Callable[[str, float], None]
RECORD_COLUMNS = ["rank", "peak_s", "timestamp", "start_s", "end_s", "score", "event_type", "reason",
                  "transcript_snippet", "clip_path", "thumbnail_path"]


@dataclass
class PipelineResult:
    highlights: list[dict]
    signals: pd.DataFrame
    reel_path: Path | None
    json_path: Path
    csv_path: Path
    out_dir: Path


def _safe_name(text: str) -> str:
    """Filename-safe version of an event type, e.g. 'Big Chance / Save' -> 'Big-Chance-Save'."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-") or "Moment"


def _cached_csv(path: Path, video: Path, use_cache: bool, compute: Callable[[], pd.DataFrame],
                keep_na: bool = True) -> pd.DataFrame:
    """Load a per-second table from the cache if valid, else compute it and save it."""
    if use_cache and is_cache_valid(path, video):
        log.info("Using cached %s", path.name)
        return pd.read_csv(path) if keep_na else pd.read_csv(path, keep_default_na=False)
    df = compute()
    df.to_csv(path, index=False)
    return df


def run_pipeline(video_path: str | Path, cfg: dict, progress_cb: ProgressCb | None = None,
                 use_cache: bool = True) -> PipelineResult:
    """Run every stage on one video and write all outputs to outputs/<video_stem>/.

    progress_cb(stage, fraction) is called with stages audio, transcript, video, scoring, clips.
    Highlights are ranked by score; the reel plays them in chronological order.
    """
    # heavy imports here so `import src.pipeline` stays cheap (e.g. for the UI)
    from .audio_extractor import extract_audio
    from .audio_features import compute_audio_features
    from .classifier import classify
    from .clipper import build_reel, build_windows, cut_clip, save_thumbnail
    from .commentary import get_snippet, score_keywords, transcribe
    from .scorer import combine_signals, find_highlights
    from .video_features import compute_video_features

    def report(stage: str, fraction: float) -> None:
        if progress_cb:
            progress_cb(stage, float(min(max(fraction, 0.0), 1.0)))

    timings: dict[str, float] = {}

    def done(stage: str, t0: float) -> None:
        timings[stage] = time.time() - t0
        log.info("Stage %-10s took %6.1f s", stage, timings[stage])
        report(stage, 1.0)

    t_all = time.time()
    video = Path(video_path).resolve()
    check_ffmpeg()
    duration = get_video_duration(video)
    n_seconds = int(duration)
    dirs = get_output_dirs(video)
    cache = dirs["cache"]
    log.info("Processing %s (%s, %d s) -> %s", video.name, fmt_time(duration), n_seconds, dirs["root"])

    # --- audio: extract wav + audio features
    t0 = time.time()
    report("audio", 0.0)
    wav = cache / "audio.wav"
    if not (use_cache and is_cache_valid(wav, video)):
        extract_audio(video, wav, cfg["audio"]["sample_rate"])
    report("audio", 0.4)
    audio_df = _cached_csv(cache / "audio_features.csv", video, use_cache,
                           lambda: compute_audio_features(wav, cfg))
    done("audio", t0)

    # --- transcript + keywords (keywords are cheap and depend on the model, so always rescored)
    t0 = time.time()
    report("transcript", 0.0)
    w = cfg["whisper"]
    transcript_cache = cache / f"transcript_{w['model']}.json"
    if not use_cache:
        transcript_cache.unlink(missing_ok=True)
    words = transcribe(wav, w["model"], w.get("backend", "auto"), transcript_cache, w.get("language", "en"))
    report("transcript", 0.9)
    kw_df = score_keywords(words, cfg["keywords"], cfg["keyword_options"], n_seconds)
    kw_df.to_csv(cache / "keywords.csv", index=False)
    done("transcript", t0)

    # --- video features
    t0 = time.time()
    report("video", 0.0)
    video_df = _cached_csv(cache / "video_features.csv", video, use_cache,
                           lambda: compute_video_features(video, cfg, lambda f: report("video", f)))
    done("video", t0)

    # --- scoring: combine -> peaks -> classify -> windows
    t0 = time.time()
    report("scoring", 0.0)
    sc = cfg["scoring"]
    signals = combine_signals(audio_df, kw_df, video_df, sc["weights"], sc["smooth_window_s"], sc)
    peaks = find_highlights(signals["score"], int(sc["top_n"]), int(sc["min_peak_distance_s"]),
                            float(sc.get("min_prominence", 0.02)))
    labels = {p: classify(p, signals, cfg) for p in peaks}
    clip_cfg = cfg.get("clips", {})
    windows = build_windows(peaks, float(clip_cfg.get("pre_s", 10)), float(clip_cfg.get("post_s", 8)), duration)
    signals_path = dirs["root"] / "signals.csv"
    signals.to_csv(signals_path, index=False)
    log.info("Found %d highlights in %d clip windows", len(peaks), len(windows))
    done("scoring", t0)

    # --- clips + thumbnails + reel
    t0 = time.time()
    report("clips", 0.0)
    for old in [*dirs["clips"].glob("*.mp4"), *dirs["thumbs"].glob("*.jpg")]:
        old.unlink()  # results of an earlier run would mix with this one
    rank_of = {p: i for i, p in enumerate(peaks, 1)}
    kw_start, kw_end = (cfg.get("classifier") or {}).get("keyword_window_s", [-5, 20])
    records: dict[int, dict] = {}
    clip_by_window: list[Path] = []
    steps = len(windows) + len(peaks) + 1
    step = 0
    for start, end, win_peaks in windows:
        best = min(win_peaks, key=rank_of.get)  # a merged window is named after its best peak
        name = f"{rank_of[best]:02d}_{fmt_time(best).replace(':', '-')}_{_safe_name(labels[best][0])}.mp4"
        clip = cut_clip(video, start, end, dirs["clips"] / name)
        clip_by_window.append(clip)
        step += 1
        report("clips", step / steps)
        for p in win_peaks:
            rank = rank_of[p]
            thumb = save_thumbnail(video, p, dirs["thumbs"] / f"{rank:02d}.jpg")
            step += 1
            report("clips", step / steps)
            event_type, reason = labels[p]
            records[rank] = {
                "rank": rank, "peak_s": int(p), "timestamp": fmt_time(p),
                "start_s": round(start, 2), "end_s": round(end, 2),
                "score": round(float(signals.loc[signals["second"] == p, "score"].iloc[0]), 4),
                "event_type": event_type, "reason": reason,
                "transcript_snippet": get_snippet(words, p + float(kw_start), p + float(kw_end)),
                "clip_path": str(clip), "thumbnail_path": str(thumb),
            }
    reel_path = build_reel(clip_by_window, dirs["root"] / "highlight_reel.mp4") if clip_by_window else None
    highlights = [records[r] for r in sorted(records)]

    json_path = dirs["root"] / "highlights.json"
    csv_path = dirs["root"] / "highlights.csv"
    json_path.write_text(json.dumps(highlights, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(highlights, columns=RECORD_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8")
    done("clips", t0)

    log.info("Done in %.1f s (%s)", time.time() - t_all,
             ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    return PipelineResult(highlights=highlights, signals=signals, reel_path=reel_path,
                          json_path=json_path, csv_path=csv_path, out_dir=dirs["root"])
