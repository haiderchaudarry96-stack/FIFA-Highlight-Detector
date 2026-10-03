"""Fuse audio, keyword and video signals into one per-second score and pick the top peaks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .utils import get_logger

log = get_logger(__name__)

# Defaults for the `scoring` config keys used here (config.yaml overrides them).
SCORING_DEFAULTS = {
    "audio_cap": 4.0,           # audio_z at/above this counts as a full crowd spike (audio_n = 1)
    "video_cap": 6.0,           # video_z at/above this = 1 (video_z is clipped at 6 anyway)
    "replay_lookahead_s": 30,   # a replay within t..t+N s marks t as "a replay follows"
    "video_mix": {"replay": 0.7, "video_z": 0.3},
    "gate_window_s": 10,        # keyword counts fully only with a crowd spike within ±N s
    "gate_floor": 0.3,          # share of the keyword signal kept with no crowd spike at all
    "min_prominence": 0.02,     # find_peaks prominence on the 0..1 score
}


def _opt(scoring_cfg: dict | None, key: str):
    """Config value with a fallback to SCORING_DEFAULTS."""
    return (scoring_cfg or {}).get(key, SCORING_DEFAULTS[key])


def _norm_weights(weights: dict) -> dict[str, float]:
    """audio/keywords/video weights, clipped to >= 0 and normalized to sum to 1."""
    w = {k: max(0.0, float((weights or {}).get(k, 0.0))) for k in ("audio", "keywords", "video")}
    total = sum(w.values())
    if total <= 0:
        raise ValueError(f"scoring.weights must have a positive sum, got {weights!r}")
    return {k: v / total for k, v in w.items()}


def _merge(audio_df, kw_df, video_df) -> pd.DataFrame:
    """Outer-join the per-second tables on `second`; missing numbers -> 0, missing hit labels -> ""."""
    frames = [df for df in (audio_df, kw_df, video_df) if df is not None and len(df.columns)]
    if not frames:
        return pd.DataFrame({"second": pd.Series(dtype=int)})
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="second", how="outer")
    merged = merged.sort_values("second").reset_index(drop=True)
    for col in merged.columns:
        if col == "keyword_hits" or merged[col].dtype == object:
            merged[col] = merged[col].fillna("").astype(str)
        else:
            merged[col] = merged[col].fillna(0)
    merged["second"] = merged["second"].astype(int)
    return merged


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column or zeros if the signal is missing."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)
    return pd.Series(0.0, index=df.index)


def combine_signals(audio_df: pd.DataFrame | None, kw_df: pd.DataFrame | None,
                    video_df: pd.DataFrame | None, weights: dict, smooth_window_s: int,
                    scoring_cfg: dict | None = None) -> pd.DataFrame:
    """Merged per-second table with the scaled signals and the final smoothed `score` (0..1).

    1. Scale to 0..1: audio_n = clip(audio_z, 0, audio_cap)/audio_cap; keyword_score is already 0..1;
       video_n = mix_replay * replay_ahead + mix_z * clip(video_z, 0, video_cap)/video_cap, where
       replay_ahead = 1 if any replay second lies in [t, t + replay_lookahead_s] (replays follow the event).
    2. Gate keywords by the crowd: keyword_gated = keyword_score * (gate_floor + (1 - gate_floor) *
       audio_peak_local), audio_peak_local = max audio_n within ±gate_window_s. Recap talk over a calm
       crowd keeps only gate_floor of its value; a live call over a crowd roar keeps all of it.
    3. score = weighted sum (weights normalized to sum 1), centered moving average of smooth_window_s.

    `scoring_cfg` is the config `scoring` section (caps, mix, gate); missing keys use SCORING_DEFAULTS.
    """
    w = _norm_weights(weights)
    df = _merge(audio_df, kw_df, video_df)
    if "keyword_hits" not in df.columns:
        df["keyword_hits"] = ""

    audio_cap = max(float(_opt(scoring_cfg, "audio_cap")), 1e-6)
    video_cap = max(float(_opt(scoring_cfg, "video_cap")), 1e-6)
    lookahead = max(0, int(_opt(scoring_cfg, "replay_lookahead_s")))
    mix = {**SCORING_DEFAULTS["video_mix"], **(_opt(scoring_cfg, "video_mix") or {})}
    mix_total = max(float(mix["replay"]) + float(mix["video_z"]), 1e-6)
    mix_replay, mix_z = float(mix["replay"]) / mix_total, float(mix["video_z"]) / mix_total
    gate_window = max(0, int(_opt(scoring_cfg, "gate_window_s")))
    gate_floor = float(np.clip(float(_opt(scoring_cfg, "gate_floor")), 0.0, 1.0))

    n = len(df)
    df["audio_n"] = _col(df, "audio_z").clip(0, audio_cap) / audio_cap
    df["keyword_n"] = _col(df, "keyword_score").clip(0, 1)

    # replay_ahead[t] = max(replay[t .. t+lookahead]): reverse, trailing rolling max, reverse back.
    replay = (_col(df, "replay") > 0).astype(float)
    df["replay_ahead"] = replay[::-1].rolling(lookahead + 1, min_periods=1).max()[::-1].to_numpy() if n else []
    df["video_n"] = (mix_replay * df["replay_ahead"]
                     + mix_z * _col(df, "video_z").clip(0, video_cap) / video_cap)

    df["audio_peak_local"] = df["audio_n"].rolling(2 * gate_window + 1, center=True, min_periods=1).max()
    df["keyword_gated"] = df["keyword_n"] * (gate_floor + (1.0 - gate_floor) * df["audio_peak_local"])

    df["score_raw"] = (w["audio"] * df["audio_n"] + w["keywords"] * df["keyword_gated"]
                       + w["video"] * df["video_n"])
    window = max(1, int(smooth_window_s))
    df["score"] = df["score_raw"].rolling(window, center=True, min_periods=1).mean()
    return df


def find_highlights(score, top_n: int, min_distance_s: int, min_prominence: float = 0.02) -> list[int]:
    """Peak seconds of `score`, highest first (ties -> earlier), at most top_n, >= min_distance_s apart.

    `score` is a Series indexed by second (as from combine_signals) or a plain array (index = second).
    The ends count as possible peaks (a goal in the last seconds of a short clip); a flat or empty
    signal gives [].
    """
    if top_n <= 0:
        return []
    if isinstance(score, pd.Series):
        seconds = np.asarray(score.index, dtype=int)
        values = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    else:
        values = np.asarray(score, dtype=float).ravel()
        seconds = np.arange(len(values))
    if values.size == 0:
        return []
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad both ends with the minimum so a maximum at the first/last second can be found.
    low = values.min()
    padded = np.concatenate(([low], values, [low]))
    idx, _ = find_peaks(padded, distance=max(1, int(min_distance_s)), prominence=max(0.0, float(min_prominence)))
    idx = idx - 1
    if idx.size == 0:
        return []
    order = sorted(idx, key=lambda i: (-values[i], i))
    return [int(seconds[i]) for i in order[:top_n]]
