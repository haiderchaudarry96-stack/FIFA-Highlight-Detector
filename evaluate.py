"""Compare predicted highlights with hand-labelled true events: precision, recall, F1, time offsets.

Usage: .venv\\Scripts\\python.exe evaluate.py --pred outputs/test_match/highlights.csv --truth samples/truth_test.csv
       [--tolerance 20]

Truth CSV: column `time` (mm:ss, h:mm:ss or plain seconds), optional column `event` (e.g. goal, save, card).
Prediction CSV: the pipeline's highlights.csv (`peak_s`, `rank`, `event_type`).
A prediction within ±tolerance seconds of a true event is correct; each true event and each prediction is
matched at most once, closest pairs first.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from src.utils import fmt_time


class EvaluationError(ValueError):
    """Bad or unreadable truth/prediction file."""


def parse_time(value) -> float:
    """'06:00' -> 360, '1:02:03' -> 3723, '95' / 95.5 -> seconds. Raises ValueError on bad input."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty time")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad time {text!r} (use mm:ss, h:mm:ss or seconds)")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"bad time {text!r} (use mm:ss, h:mm:ss or seconds)") from None
    if any(math.isnan(n) or n < 0 for n in nums) or (len(nums) > 1 and nums[-1] >= 60) \
            or (len(nums) == 3 and nums[1] >= 60):
        raise ValueError(f"bad time {text!r} (use mm:ss, h:mm:ss or seconds)")
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV as strings with trimmed headers/cells; an empty file gives an empty table."""
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"File not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, skipinitialspace=True, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.apply(lambda col: col.str.strip())


def load_truth(path: str | Path) -> list[dict]:
    """True events as [{'time_s': float, 'event': str}] sorted by time (event '' if there is no event column)."""
    df = _read_csv(path)
    if df.empty:
        return []
    if "time" not in df.columns:
        raise EvaluationError(f"{path}: needs a 'time' column (found: {', '.join(df.columns)})")
    events = []
    for i, row in df.iterrows():
        try:
            t = parse_time(row["time"])
        except ValueError as e:
            raise EvaluationError(f"{path}, row {i + 2}: {e}") from None
        events.append({"time_s": t, "event": row.get("event", "").lower()})
    return sorted(events, key=lambda e: e["time_s"])


def load_predictions(path: str | Path) -> list[dict]:
    """Predictions as [{'rank', 'peak_s', 'event_type'}] from a highlights.csv."""
    df = _read_csv(path)
    if df.empty:
        return []
    if "peak_s" not in df.columns:
        raise EvaluationError(f"{path}: needs a 'peak_s' column (is this a highlights.csv?)")
    preds = []
    for i, row in df.iterrows():
        try:
            peak = float(row["peak_s"])
            rank = int(float(row["rank"])) if row.get("rank", "") else i + 1
        except ValueError:
            raise EvaluationError(f"{path}, row {i + 2}: bad number in peak_s/rank") from None
        preds.append({"rank": rank, "peak_s": peak, "event_type": row.get("event_type", "")})
    return preds


def match_events(truth: list[dict], preds: list[dict], tolerance: float = 20.0) -> dict:
    """Greedy one-to-one matching, closest (truth, prediction) pairs first; ties go to the better rank.

    Returns {'matches': [(truth, pred, offset_s)], 'missed': [truth], 'false_positives': [pred]},
    offset = predicted - true time.
    """
    pairs = sorted(
        ((abs(p["peak_s"] - t["time_s"]), p["rank"], ti, pi)
         for ti, t in enumerate(truth) for pi, p in enumerate(preds)
         if abs(p["peak_s"] - t["time_s"]) <= tolerance))
    used_t, used_p, matches = set(), set(), []
    for _, _, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matches.append((truth[ti], preds[pi], preds[pi]["peak_s"] - truth[ti]["time_s"]))
    matches.sort(key=lambda m: m[0]["time_s"])
    return {
        "matches": matches,
        "missed": [t for i, t in enumerate(truth) if i not in used_t],
        "false_positives": sorted((p for i, p in enumerate(preds) if i not in used_p), key=lambda p: p["rank"]),
    }


def compute_metrics(truth: list[dict], preds: list[dict], tolerance: float = 20.0) -> dict:
    """Counts, precision/recall/F1 (None when undefined), mean |offset| and the goal-only count."""
    m = match_events(truth, preds, tolerance)
    tp = len(m["matches"])
    precision = tp / len(preds) if preds else None
    recall = tp / len(truth) if truth else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall > 0 else
          (0.0 if precision is not None and recall is not None else None))
    offsets = [abs(o) for _, _, o in m["matches"]]
    has_events = any(t["event"] for t in truth)
    goals = [t for t in truth if t["event"] == "goal"] if has_events else list(truth)
    goal_ids = {id(t) for t in goals}
    goals_found = sum(1 for t, p, _ in m["matches"] if id(t) in goal_ids and p["event_type"] == "Goal")
    return {**m, "n_pred": len(preds), "n_truth": len(truth), "tp": tp, "precision": precision,
            "recall": recall, "f1": f1, "mean_abs_offset": sum(offsets) / len(offsets) if offsets else None,
            "n_goals": len(goals), "goals_found": goals_found, "goals_from_event_column": has_events}


def format_report(r: dict, tolerance: float) -> str:
    """Human-readable report of compute_metrics output."""
    def num(x, fmt="{:.3f}"):
        return "n/a" if x is None else fmt.format(x)

    lines = [
        f"Predictions:      {r['n_pred']}",
        f"True events:      {r['n_truth']}",
        f"True positives:   {r['tp']}   (tolerance +/-{tolerance:g} s)",
        f"Precision:        {num(r['precision'])}",
        f"Recall:           {num(r['recall'])}",
        f"F1:               {num(r['f1'])}",
        f"Avg |offset|:     {num(r['mean_abs_offset'], '{:.1f} s')}",
        f"Goal-only:        {r['goals_found']} of {r['n_goals']} true "
        f"{'goals' if r['goals_from_event_column'] else 'events (no event column)'} found AND labelled 'Goal'",
        "",
        "Matched (true -> predicted):",
    ]
    if r["matches"]:
        lines.append(f"  {'true':>7}  {'pred':>7}  {'offset':>7}  {'rank':>4}  label")
        for t, p, off in r["matches"]:
            lines.append(f"  {fmt_time(t['time_s']):>7}  {fmt_time(p['peak_s']):>7}  {off:>+6.0f}s  "
                         f"{p['rank']:>4}  {p['event_type']}")
    else:
        lines.append("  (none)")
    lines.append("Missed true events:")
    lines += [f"  {fmt_time(t['time_s']):>7}  {t['event']}" for t in r["missed"]] or ["  (none)"]
    lines.append("False positives:")
    if r["false_positives"]:
        lines.append(f"  {'time':>7}  {'rank':>4}  label")
        lines += [f"  {fmt_time(p['peak_s']):>7}  {p['rank']:>4}  {p['event_type']}" for p in r["false_positives"]]
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate predicted highlights against true event times.")
    ap.add_argument("--pred", required=True, type=Path, help="highlights.csv from the pipeline")
    ap.add_argument("--truth", required=True, type=Path, help="truth CSV with a 'time' column (+ optional 'event')")
    ap.add_argument("--tolerance", type=float, default=20.0, help="max |predicted - true| in seconds (default 20)")
    args = ap.parse_args()
    if args.tolerance < 0:
        print("Error: --tolerance must be >= 0", file=sys.stderr)
        return 1
    try:
        truth = load_truth(args.truth)
        preds = load_predictions(args.pred)
    except EvaluationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not truth:
        print(f"Warning: {args.truth} has no true events; recall is undefined.", file=sys.stderr)
    if not preds:
        print(f"Warning: {args.pred} has no predictions; precision is undefined.", file=sys.stderr)
    print(f"Evaluating {args.pred} against {args.truth}\n")
    print(format_report(compute_metrics(truth, preds, args.tolerance), args.tolerance))
    return 0


if __name__ == "__main__":
    sys.exit(main())
