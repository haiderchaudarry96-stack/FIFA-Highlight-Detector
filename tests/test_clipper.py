"""clipper.build_windows tests (pure logic, no ffmpeg)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clipper import build_windows  # noqa: E402


def test_separate_windows_stay_apart_and_sorted():
    w = build_windows([200, 50], pre_s=10, post_s=8, duration=600)
    assert w == [(40.0, 58.0, [50]), (190.0, 208.0, [200])]


def test_overlapping_windows_merge_and_keep_peaks():
    # 100 -> [90, 108], 105 -> [95, 113], 150 -> [140, 158]
    w = build_windows([105, 100, 150], pre_s=10, post_s=8, duration=600)
    assert w == [(90.0, 113.0, [100, 105]), (140.0, 158.0, [150])]


def test_chain_of_overlaps_merges_into_one():
    w = build_windows([10, 25, 40], pre_s=10, post_s=8, duration=600)
    assert w == [(0.0, 48.0, [10, 25, 40])]


def test_touching_windows_merge():
    # 100 -> [90, 108], 118 -> [108, 126]
    w = build_windows([100, 118], pre_s=10, post_s=8, duration=600)
    assert w == [(90.0, 126.0, [100, 118])]


def test_clamp_at_zero():
    w = build_windows([3], pre_s=10, post_s=8, duration=600)
    assert w == [(0.0, 11.0, [3])]


def test_clamp_at_duration():
    w = build_windows([595], pre_s=10, post_s=8, duration=600.5)
    assert w == [(585.0, 600.5, [595])]


def test_empty():
    assert build_windows([], pre_s=10, post_s=8, duration=600) == []
