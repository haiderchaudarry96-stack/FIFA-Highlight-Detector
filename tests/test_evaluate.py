"""Unit tests for evaluate.py (pure functions + small temp CSVs, no video)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import (EvaluationError, compute_metrics, load_predictions, load_truth,  # noqa: E402
                      match_events, parse_time)


def truth(*times, event="goal"):
    return [{"time_s": float(t), "event": event} for t in times]


def preds(*items):
    """items: (peak_s, event_type) in rank order."""
    return [{"rank": i, "peak_s": float(t), "event_type": e} for i, (t, e) in enumerate(items, 1)]


@pytest.mark.parametrize("text, expected", [
    ("06:00", 360), ("6:00", 360), ("95:03", 5703), ("1:02:03", 3723), ("75", 75), ("75.5", 75.5), (" 00:07 ", 7),
])
def test_parse_time(text, expected):
    assert parse_time(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "6:75", "1:60:00", "-5", "1:2:3:4", "6-00"])
def test_parse_time_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_time(bad)


def test_perfect_match():
    r = compute_metrics(truth(360, 1534), preds((361, "Goal"), (1530, "Goal")))
    assert (r["tp"], r["precision"], r["recall"], r["f1"]) == (2, 1.0, 1.0, 1.0)
    assert r["mean_abs_offset"] == pytest.approx(2.5)
    assert r["goals_found"] == 2 and not r["missed"] and not r["false_positives"]


def test_one_miss():
    r = compute_metrics(truth(360, 1534), preds((361, "Goal")))
    assert r["tp"] == 1 and r["recall"] == 0.5 and r["precision"] == 1.0
    assert [t["time_s"] for t in r["missed"]] == [1534]


def test_one_false_positive_and_outside_tolerance():
    r = compute_metrics(truth(360), preds((361, "Goal"), (900, "Card / Foul"), (381, "Goal")), tolerance=20)
    assert r["tp"] == 1 and r["precision"] == pytest.approx(1 / 3)
    assert [p["peak_s"] for p in r["false_positives"]] == [900, 381]  # 381 is 21 s away


def test_duplicate_predictions_near_one_event():
    # two predictions near one goal: the closer one matches, the other is a false positive
    m = match_events(truth(360), preds((370, "Exciting Moment"), (358, "Goal")))
    assert len(m["matches"]) == 1 and m["matches"][0][1]["peak_s"] == 358
    assert [p["peak_s"] for p in m["false_positives"]] == [370]


def test_nearest_pairs_win_across_events():
    # 345 is within tolerance of both events; it goes to the closer one so both get matched
    m = match_events(truth(340, 360), preds((345, "Goal"), (365, "Goal")))
    assert sorted((t["time_s"], p["peak_s"]) for t, p, _ in m["matches"]) == [(340, 345), (360, 365)]


def test_goal_only_needs_goal_label():
    r = compute_metrics(truth(360) + truth(800, event="save"), preds((361, "Card / Foul"), (801, "Goal")))
    assert r["tp"] == 2 and r["n_goals"] == 1 and r["goals_found"] == 0


def test_empty_inputs():
    r = compute_metrics([], [])
    assert r["tp"] == 0 and r["precision"] is None and r["recall"] is None and r["f1"] is None
    r = compute_metrics(truth(360), [])
    assert r["recall"] == 0.0 and r["precision"] is None and len(r["missed"]) == 1
    r = compute_metrics([], preds((10, "Goal")))
    assert r["precision"] == 0.0 and r["recall"] is None


def test_load_files(tmp_path):
    t = tmp_path / "truth.csv"
    t.write_text("   time,event\n   06:00,goal\n   1534,Goal\n", encoding="utf-8")  # leading spaces, mixed case
    assert load_truth(t) == [{"time_s": 360.0, "event": "goal"}, {"time_s": 1534.0, "event": "goal"}]
    no_event = tmp_path / "times.csv"
    no_event.write_text("time\n1:00\n", encoding="utf-8")
    assert load_truth(no_event) == [{"time_s": 60.0, "event": ""}]
    p = tmp_path / "highlights.csv"
    p.write_text("rank,peak_s,timestamp,event_type\n1,361,06:01,Goal\n", encoding="utf-8")
    assert load_predictions(p) == [{"rank": 1, "peak_s": 361.0, "event_type": "Goal"}]
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert load_truth(empty) == [] and load_predictions(empty) == []
    header_only = tmp_path / "header.csv"
    header_only.write_text("rank,peak_s,event_type\n", encoding="utf-8")
    assert load_predictions(header_only) == []


def test_load_errors(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("time,event\n6:75,goal\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="row 2"):
        load_truth(bad)
    no_time = tmp_path / "no_time.csv"
    no_time.write_text("minute,event\n6,goal\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="'time' column"):
        load_truth(no_time)
    with pytest.raises(EvaluationError, match="peak_s"):
        load_predictions(no_time)
    with pytest.raises(EvaluationError, match="not found"):
        load_truth(tmp_path / "missing.csv")
