from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detect_space_anomalies import (  # noqa: E402
    _compute_recovery_minutes,
    build_event_baseline_summaries,
    classify_severity_tier,
)


class TestClassifySeverityTier:
    @pytest.mark.parametrize(
        ("max_z", "synced_count", "expected"),
        [
            (6.0, 4, "CRITICAL"),
            (7.88, 5, "CRITICAL"),
            (4.0, 3, "HIGH"),
            (5.9, 3, "HIGH"),
            (6.0, 2, "HIGH"),
            (7.5, 2, "HIGH"),
            (3.0, 2, "MODERATE"),
            (3.99, 3, "MODERATE"),
            (5.99, 2, "MODERATE"),
            (2.99, 2, "LOW"),
            (4.5, 1, "LOW"),
            (1.5, 0, "LOW"),
        ],
    )
    def test_severity_tier_rules_and_boundaries(
        self, max_z: float, synced_count: int, expected: str
    ) -> None:
        assert classify_severity_tier(max_z, synced_count) == expected


class TestComputeRecoveryMinutes:
    def test_requires_consecutive_streak_and_resets_on_spike(self) -> None:
        event_end = pd.Timestamp("2026-11-21T12:00:00Z")
        base_val = 400.0
        base_scale = 10.0  # tolerance = 0.75 * 10.0 = 7.5

        post_df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2026-11-21T12:01:00Z"),
                    pd.Timestamp("2026-11-21T12:02:00Z"),
                    pd.Timestamp("2026-11-21T12:03:00Z"),
                    pd.Timestamp("2026-11-21T12:04:00Z"),
                    pd.Timestamp("2026-11-21T12:05:00Z"),
                    pd.Timestamp("2026-11-21T12:06:00Z"),
                ],
                "proton_speed": [402.0, 404.0, 420.0, 403.0, 398.0, 401.0],
            }
        )

        rec = _compute_recovery_minutes(
            post_slice=post_df,
            col="proton_speed",
            event_end=event_end,
            base_val=base_val,
            base_scale=base_scale,
            recovery_z_tol=0.75,
            min_recovery_streak=3,
        )
        assert rec == 4.0

    def test_returns_none_when_post_slice_empty_or_streak_not_met(self) -> None:
        event_end = pd.Timestamp("2026-11-21T12:00:00Z")
        empty_post = pd.DataFrame(
            {"timestamp": pd.Series([], dtype="datetime64[ns, UTC]"), "bt": []}
        )

        assert (
            _compute_recovery_minutes(
                post_slice=empty_post,
                col="bt",
                event_end=event_end,
                base_val=5.0,
                base_scale=1.0,
                min_recovery_streak=3,
            )
            is None
        )

        partial_post = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2026-11-21T12:01:00Z"),
                    pd.Timestamp("2026-11-21T12:02:00Z"),
                    pd.Timestamp("2026-11-21T12:03:00Z"),
                ],
                "bt": [5.1, 5.2, 9.5],
            }
        )
        assert (
            _compute_recovery_minutes(
                post_slice=partial_post,
                col="bt",
                event_end=event_end,
                base_val=5.0,
                base_scale=1.0,
                recovery_z_tol=0.75,
                min_recovery_streak=3,
            )
            is None
        )


class TestBuildEventBaselineSummaries:
    def test_pre_baseline_tail_cutoff_and_event_consolidation(self) -> None:
        n_rows = 60
        timestamps = pd.date_range("2026-11-21T09:00:00Z", periods=n_rows, freq="1min")
        metric_a = [10.0] * n_rows
        metric_b = [5.0] * n_rows
        synced_event = [False] * n_rows
        hit_a = [False] * n_rows
        hit_b = [False] * n_rows
        max_z = [0.5] * n_rows

        # Pre-baseline window before Event #1 (indices 0..9): metric_a = 10.0, metric_b = 5.0
        # Event #1 (indices 10..11): 09:10Z ~ 09:11Z
        for idx in (10, 11):
            metric_a[idx] = 30.0
            metric_b[idx] = 15.0
            synced_event[idx] = True
            hit_a[idx] = True
            hit_b[idx] = True
            max_z[idx] = 6.5

        # Event #2 at tail (index 59 -> 09:59Z, 48 min after 09:11Z > 20min gap_tolerance)
        tail_idx = n_rows - 1
        metric_a[tail_idx] = 22.0
        metric_b[tail_idx] = 11.0
        synced_event[tail_idx] = True
        hit_a[tail_idx] = True
        hit_b[tail_idx] = True
        max_z[tail_idx] = 3.5

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric_a": metric_a,
                "metric_b": metric_b,
                "sync_hit_metric_a": hit_a,
                "sync_hit_metric_b": hit_b,
                "max_metric_z": max_z,
                "synced_event": synced_event,
            }
        )

        summaries = build_event_baseline_summaries(
            df=df,
            metric_cols=["metric_a", "metric_b"],
            pre_window=pd.Timedelta("10min"),
            post_window=pd.Timedelta("10min"),
            gap_tolerance=pd.Timedelta("20min"),
            recovery_z_tol=0.75,
            min_recovery_streak=3,
        )

        assert len(summaries) == 2

        ev1 = summaries[0]
        assert ev1.event_id == 1
        assert ev1.severity_tier == "HIGH"
        assert ev1.synced_metrics == ["metric_a", "metric_b"]
        assert ev1.pre_baseline_mean["metric_a"] == pytest.approx(10.0)
        assert ev1.pre_to_peak_delta["metric_a"] == pytest.approx(20.0)
        assert ev1.post_samples_count > 0

        ev2 = summaries[1]
        assert ev2.event_id == 2
        assert ev2.severity_tier == "MODERATE"
        assert ev2.post_samples_count == 0
        assert ev2.recovery_minutes["metric_a"] is None
        assert ev2.recovery_minutes["metric_b"] is None