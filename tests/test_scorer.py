"""Scorer + classifier tests on synthetic numpy signals (no real video)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import classify  # noqa: E402
from src.scorer import combine_signals, find_highlights  # noqa: E402

N = 300
W = {"audio": 0.5, "keywords": 0.35, "video": 0.15}
SCORING = {"audio_cap": 4.0, "video_cap": 6.0, "replay_lookahead_s": 30,
           "video_mix": {"replay": 0.7, "video_z": 0.3}, "gate_window_s": 10, "gate_floor": 0.3}
CFG = {
    "classifier": {"keyword_window_s": [-5, 20], "replay_window_s": [0, 60], "goal_audio_z_min": 1.0,
                   "confirm_keywords": ["scores!", "scored!", "N-N", "takes the lead"]},
    "keywords": {"goal": 1.0, "scores": 0.9, "scored": 0.8, "save": 0.7, "penalty": 0.8, "it's 1-0": 1.0},
    "keyword_options": {"elongation_boost": 0.5, "exclamation_boost": 0.5, "repeat_boost": 0.5},
}


def tables(n=N, audio=(), kw=(), replay=(), video=()):
    """Per-second audio/keyword/video tables; each spec is (second, value) [or (second, value, label)]."""
    sec = np.arange(n)
    a = pd.DataFrame({"second": sec, "audio_z": 0.0})
    k = pd.DataFrame({"second": sec, "keyword_score": 0.0, "keyword_hits": ""})
    v = pd.DataFrame({"second": sec, "replay": 0, "video_z": 0.0})
    for s, val in audio:
        a.loc[s, "audio_z"] = val
    for s, val, *label in kw:
        k.loc[s, "keyword_score"] = val
        if label:
            k.loc[s, "keyword_hits"] = label[0]
    for s in replay:
        v.loc[s, "replay"] = 1
    for s, val in video:
        v.loc[s, "video_z"] = val
    return a, k, v


def spike(center, width=3, height=5.0):
    return [(s, height) for s in range(center - width, center + width + 1)]


def score_of(*args, weights=W, smooth=5, **kw):
    return combine_signals(*tables(*args, **kw), weights, smooth, SCORING)


# ---------------------------------------------------------------- combine_signals

def test_columns_and_range():
    df = score_of(audio=spike(100), kw=[(100, 1.0, "goal(1.00)")], replay=[120])
    for col in ("second", "audio_n", "keyword_gated", "video_n", "score", "keyword_hits"):
        assert col in df.columns
    assert len(df) == N
    assert df["score"].between(0, 1).all()


def test_outer_join_fills_missing_seconds():
    a, k, v = tables(50)
    df = combine_signals(a.iloc[:40], k.iloc[10:], v, W, 1, SCORING)
    assert df["second"].tolist() == list(range(50))
    assert not df[["audio_n", "keyword_gated", "video_n", "score"]].isna().any().any()
    assert (df["keyword_hits"] == "").all()


# ---------------------------------------------------------------- find_highlights

def test_two_clear_spikes_give_two_peaks():
    df = score_of(audio=spike(80) + spike(220))
    peaks = find_highlights(df["score"], top_n=10, min_distance_s=30)
    assert len(peaks) == 2
    assert {round(p, -1) for p in peaks} == {80, 220}


def test_close_spikes_merge_with_min_distance():
    df = score_of(audio=spike(100, width=1) + spike(110, width=1, height=3.0), smooth=1)
    assert len(find_highlights(df["score"], top_n=10, min_distance_s=5)) == 2
    peaks = find_highlights(df["score"], top_n=10, min_distance_s=30)
    assert peaks == [100]   # the higher one wins


def test_flat_and_empty_signals():
    assert find_highlights(np.zeros(100), 10, 30) == []
    assert find_highlights(np.full(100, 0.4), 10, 30) == []
    assert find_highlights(np.array([]), 10, 30) == []
    assert find_highlights(pd.Series(dtype=float), 10, 30) == []
    df = combine_signals(*tables(0), W, 5, SCORING)
    assert len(df) == 0 and find_highlights(df["score"], 10, 30) == []


def test_fewer_peaks_than_top_n_and_top_n_limit():
    df = score_of(audio=spike(50) + spike(150, height=3.0) + spike(250, height=2.0))
    assert len(find_highlights(df["score"], top_n=10, min_distance_s=30)) == 3
    assert find_highlights(df["score"], top_n=1, min_distance_s=30) == [50]


def test_very_short_video_and_edge_peak():
    assert find_highlights(np.array([0.1, 0.2, 0.9]), 5, 30) == [2]
    assert find_highlights(np.array([0.5]), 5, 30) == []


def test_weights_change_ranking():
    # crowd roar at 60, keyword call (no roar) at 200
    args = dict(audio=spike(60), kw=[(s, 1.0) for s in range(197, 204)])
    audio_heavy = score_of(**args, weights={"audio": 0.8, "keywords": 0.2, "video": 0.0})
    kw_heavy = score_of(**args, weights={"audio": 0.05, "keywords": 0.95, "video": 0.0})
    assert find_highlights(audio_heavy["score"], 2, 30)[0] == 60
    assert find_highlights(kw_heavy["score"], 2, 30)[0] == 200


def test_keyword_gated_by_audio():
    kw = [(s, 1.0) for s in range(98, 103)]
    with_crowd = score_of(audio=spike(100), kw=kw)
    no_crowd = score_of(kw=kw)
    assert with_crowd.loc[100, "keyword_gated"] > no_crowd.loc[100, "keyword_gated"]
    assert no_crowd.loc[100, "keyword_gated"] > 0          # the gate floor keeps a little
    assert np.isclose(no_crowd.loc[100, "keyword_gated"], 0.3)


def test_replay_after_event_raises_event_not_replay():
    event, replay = 100, list(range(115, 135))
    base = score_of(audio=spike(event))
    with_replay = score_of(audio=spike(event), replay=replay)
    assert with_replay.loc[event, "score"] > base.loc[event, "score"]
    assert with_replay.loc[event, "video_n"] > 0
    assert with_replay.loc[event - 40, "video_n"] == 0     # a replay never looks backwards in time
    peaks = find_highlights(with_replay["score"], 5, 30)
    assert peaks[0] == event


# ---------------------------------------------------------------- classify

def test_classify_live_call_is_goal():
    # "Costa scores!" (0.9 x 1.5 exclamation) with only a moderate crowd and no replay
    df = score_of(audio=spike(100, height=1.9), kw=[(101, 1.0, "scores(1.35)"), (106, 0.5, "goal(1.00)")])
    label, reason = classify(100, df, CFG)
    assert label == "Goal"
    assert "confirmed by 'scores!'" in reason and "'goal'" in reason and "audio z=1.9" in reason
    assert "replay" not in reason


def test_classify_scoreline_goal_mentions_replay():
    df = score_of(audio=spike(100, height=3.0), kw=[(102, 1.0, "it's 1-0(1.00)")], replay=range(110, 130))
    label, reason = classify(100, df, CFG)
    assert label == "Goal"
    assert "confirmed by 'it's 1-0'" in reason and "replay detected" in reason


def test_classify_near_miss_is_not_goal():
    # loud crowd, 'goal' and 'penalty' talk, a replay, but no confirm keyword
    df = score_of(audio=spike(100, height=6.0), kw=[(103, 1.0, "goal(1.50)"), (108, 0.8, "penalty(0.80)")],
                  replay=range(115, 140))
    label, reason = classify(100, df, CFG)
    assert label != "Goal"
    assert "replay detected" in reason and "confirmed" not in reason


def test_classify_unexclaimed_scored_is_not_goal():
    # calm "... he scored" recap: "scored!" needs the exclamation boost
    df = score_of(audio=spike(100, height=3.0), kw=[(102, 1.0, "scored(0.80)")])
    assert classify(100, df, CFG)[0] != "Goal"


def test_classify_recap_is_not_goal():
    # live-call words but a calm crowd, no replay -> below the audio floor
    df = score_of(kw=[(100, 1.0, "goal(1.00)"), (104, 1.0, "scores(1.35)")])
    label, reason = classify(100, df, CFG)
    assert label != "Goal"
    assert "calm crowd (audio z=0.0)" in reason


def test_classify_other_types():
    df = score_of(audio=spike(100), kw=[(102, 1.0, "save(0.70)")])
    assert classify(100, df, CFG)[0] == "Big Chance / Save"
    df = score_of(audio=spike(100), kw=[(102, 1.0, "penalty(0.80)")])
    assert classify(100, df, CFG)[0] == "Card / Foul"
    df = score_of(audio=spike(100))
    assert classify(100, df, CFG)[0] == "Exciting Moment"
