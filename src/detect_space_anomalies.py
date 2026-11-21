from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

NOAA_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
NOAA_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"


@dataclass
class EventBaselineSummary:
    event_id: int
    start_time: str
    end_time: str
    duration_minutes: float
    synced_metrics_count: int
    synced_metrics: list[str]
    max_composite_z: float
    severity_tier: str
    post_samples_count: int
    recovery_streak_required: int
    pre_baseline_mean: dict[str, float]
    peak_values: dict[str, float]
    post_window_mean: dict[str, float]
    pre_to_peak_delta: dict[str, float]
    post_to_pre_delta: dict[str, float]
    recovery_minutes: dict[str, float | None]


def classify_severity_tier(max_composite_z: float, synced_metrics_count: int) -> str:
    """依 max_composite_z 與 synced_metrics_count 套用確定性事件分級規則。"""
    if max_composite_z >= 6.0 and synced_metrics_count >= 4:
        return "CRITICAL"
    if (max_composite_z >= 4.0 and synced_metrics_count >= 3) or (
        max_composite_z >= 6.0 and synced_metrics_count >= 2
    ):
        return "HIGH"
    if max_composite_z >= 3.0 and synced_metrics_count >= 2:
        return "MODERATE"
    return "LOW"


def _fetch_json(url: str, timeout: int = 10) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "SpaceAnomalyMVP/0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Unexpected or empty JSON from {url}")
    return payload


def fetch_merged_space_weather_df(
    mag_url: str = NOAA_MAG_URL,
    wind_url: str = NOAA_WIND_URL,
    tail_rows: int = 1500,
) -> pd.DataFrame:
    """合併磁場與太陽風公開資料；若無法連線或欄位不符，退回合成多維度資料。"""
    try:
        mag_raw = _fetch_json(mag_url)
        wind_raw = _fetch_json(wind_url)

        mag_df = pd.DataFrame(mag_raw)
        wind_df = pd.DataFrame(wind_raw)

        mag_time = "time_tag" if "time_tag" in mag_df.columns else mag_df.columns[0]
        wind_time = "time_tag" if "time_tag" in wind_df.columns else wind_df.columns[0]

        mag_df["timestamp"] = pd.to_datetime(mag_df[mag_time], utc=True, errors="coerce")
        wind_df["timestamp"] = pd.to_datetime(wind_df[wind_time], utc=True, errors="coerce")

        mag_cols = [c for c in ["bx_gsm", "by_gsm", "bz_gsm", "bt"] if c in mag_df.columns]
        wind_cols = [c for c in ["proton_speed", "proton_density"] if c in wind_df.columns]
        if not mag_cols or not wind_cols:
            raise ValueError("Missing expected magnetic or wind columns")

        for col in mag_cols:
            mag_df[col] = pd.to_numeric(mag_df[col], errors="coerce")
        for col in wind_cols:
            wind_df[col] = pd.to_numeric(wind_df[col], errors="coerce")

        m_clean = mag_df[["timestamp", *mag_cols]].dropna().sort_values("timestamp")
        w_clean = wind_df[["timestamp", *wind_cols]].dropna().sort_values("timestamp")

        merged = pd.merge_asof(
            m_clean,
            w_clean,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("2min"),
        ).dropna()

        if merged.empty:
            raise ValueError("Merged DataFrame is empty")
        return merged.tail(tail_rows).reset_index(drop=True)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        print(f"[Info] Using synthetic multi-metric dataset ({exc})")
        return build_synthetic_multimetric_df()


def build_synthetic_multimetric_df(n_rows: int = 720) -> pd.DataFrame:
    """產生含『單一指標雜訊』與『多指標同步事件＋前後基線變化與恢復曲線』的測試資料。"""
    rng = pd.date_range("2026-11-18T00:00:00Z", periods=n_rows, freq="10min")
    bt = [5.2 + ((i % 7) - 3) * 0.3 for i in range(n_rows)]
    bz_gsm = [-1.2 + ((i % 9) - 4) * 0.35 for i in range(n_rows)]
    proton_speed = [410.0 + ((i % 11) - 5) * 8.0 for i in range(n_rows)]
    proton_density = [6.0 + ((i % 5) - 2) * 0.5 for i in range(n_rows)]

    for idx in range(180, 186):
        bt[idx] = 21.0
        bz_gsm[idx] = -16.5
        proton_speed[idx] = 660.0
        proton_density[idx] = 18.0

    for offset, idx in enumerate(range(186, 200)):
        decay = max(0.0, 1.0 - offset / 14.0)
        bt[idx] = 5.2 + 7.5 * decay
        bz_gsm[idx] = -1.2 - 6.0 * decay
        proton_speed[idx] = 410.0 + 115.0 * decay
        proton_density[idx] = 6.0 + 5.0 * decay

    proton_density[450] = 28.0

    return pd.DataFrame(
        {
            "timestamp": rng,
            "bt": bt,
            "bz_gsm": bz_gsm,
            "proton_speed": proton_speed,
            "proton_density": proton_density,
        }
    )


def _robust_scale(series: pd.Series, fallback_series: pd.Series | None = None) -> float:
    """以滾動／事前視窗的 MAD 換算穩健標準差；若離散度為 0 則退回標準差。"""
    s = series.dropna()
    if len(s) >= 5:
        median = float(s.median())
        mad = float((s - median).abs().median())
        mad_sigma = 1.4826 * mad
        if mad_sigma > 1e-9:
            return mad_sigma

    std = float(s.std(ddof=0)) if len(s) >= 2 else 0.0
    if std > 1e-9:
        return std

    if fallback_series is not None and not fallback_series.empty:
        fb_std = float(fallback_series.dropna().std(ddof=0))
        if fb_std > 1e-9:
            return fb_std

    return 0.0


def detect_synced_anomalies(
    df: pd.DataFrame,
    sync_window: str = "30min",
    baseline_window: str = "2h",
    z_threshold: float = 2.5,
    min_synced_metrics: int = 2,
    contamination: float = 0.03,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """以事件前滾動基線計算偏離度與 IsolationForest 分數，判定多維度跨指標同步事件。"""
    metric_cols = [c for c in df.columns if c != "timestamp"]
    if len(metric_cols) < 2:
        raise ValueError("At least 2 numeric metrics are required for multi-metric sync detection")

    work = df.sort_values("timestamp").reset_index(drop=True).copy()
    work = work.set_index("timestamp")

    z_cols: list[str] = []
    metric_hit_cols: list[str] = []
    min_hist_periods = 10

    for col in metric_cols:
        hist_median = work[col].rolling(baseline_window, min_periods=min_hist_periods).median()
        hist_mad = (
            (work[col] - hist_median)
            .abs()
            .rolling(baseline_window, min_periods=min_hist_periods)
            .median()
            * 1.4826
        )
        hist_std = work[col].rolling(baseline_window, min_periods=min_hist_periods).std(ddof=0)

        global_median = float(work[col].median())
        global_scale = _robust_scale(work[col])

        ref_center = hist_median.fillna(global_median)
        ref_scale = hist_mad.where(hist_mad > 1e-9, hist_std).fillna(global_scale)

        z_col = f"z_{col}"
        work[z_col] = np.where(
            ref_scale > 1e-9,
            (work[col] - ref_center).abs() / ref_scale,
            0.0,
        )
        z_cols.append(z_col)

        hit_col = f"sync_hit_{col}"
        work[hit_col] = (
            work[z_col].rolling(sync_window, min_periods=1).max() >= z_threshold
        )
        metric_hit_cols.append(hit_col)

    work["synced_metrics_count"] = work[metric_hit_cols].sum(axis=1)
    work["max_metric_z"] = work[z_cols].max(axis=1)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    preds = model.fit_predict(work[metric_cols])
    work["is_isolation_anomaly"] = preds == -1
    work["anomaly_severity"] = -model.decision_function(work[metric_cols])

    work["synced_event"] = (
        (work["synced_metrics_count"] >= min_synced_metrics) & work["is_isolation_anomaly"]
    )

    out = work.reset_index()
    return out, metric_cols


def _compute_recovery_minutes(
    post_slice: pd.DataFrame,
    col: str,
    event_end: pd.Timestamp,
    base_val: float,
    base_scale: float,
    recovery_z_tol: float = 0.75,
    min_recovery_streak: int = 3,
) -> float | None:
    """
    計算事件結束後，該指標首次「連續 min_recovery_streak 筆」落在事前基線容忍帶內所需的分鐘數；
    若事後資料不足、未達連續筆數或未恢復，則回傳 None。
    """
    streak_req = max(int(min_recovery_streak), 1)
    if post_slice.empty or len(post_slice) < streak_req or col not in post_slice.columns:
        return None

    ordered_post = post_slice.sort_values("timestamp").reset_index(drop=True)
    tolerance = (
        max(recovery_z_tol * base_scale, 1e-6)
        if pd.notna(base_scale) and base_scale > 1e-9
        else 1e-6
    )

    in_band = ((ordered_post[col] - base_val).abs() <= tolerance).to_numpy(dtype=bool)
    consecutive_count = 0

    for idx, flag in enumerate(in_band):
        if flag:
            consecutive_count += 1
            if consecutive_count >= streak_req:
                streak_start_idx = idx - streak_req + 1
                first_recovered_ts = ordered_post.loc[streak_start_idx, "timestamp"]
                elapsed_min = (first_recovered_ts - event_end).total_seconds() / 60.0
                return round(max(elapsed_min, 0.0), 1)
        else:
            consecutive_count = 0

    return None


def build_event_baseline_summaries(
    df: pd.DataFrame,
    metric_cols: list[str],
    pre_window: pd.Timedelta = pd.Timedelta("2h"),
    post_window: pd.Timedelta = pd.Timedelta("2h"),
    gap_tolerance: pd.Timedelta = pd.Timedelta("20min"),
    recovery_z_tol: float = 0.75,
    min_recovery_streak: int = 3,
) -> list[EventBaselineSummary]:
    """將相鄰 synced_event 合併為事件區段，以 pre_window 計算事前基線，並以連續達標計算恢復時間。"""
    flagged = df[df["synced_event"]].sort_values("timestamp").reset_index(drop=True)
    if flagged.empty:
        return []

    segments: list[list[pd.Timestamp]] = []
    current_start = flagged.loc[0, "timestamp"]
    current_end = flagged.loc[0, "timestamp"]

    for idx in range(1, len(flagged)):
        ts = flagged.loc[idx, "timestamp"]
        if ts - current_end <= gap_tolerance:
            current_end = ts
        else:
            segments.append([current_start, current_end])
            current_start = ts
            current_end = ts
    segments.append([current_start, current_end])

    summaries: list[EventBaselineSummary] = []
    ts_series = df["timestamp"]
    streak_req = max(int(min_recovery_streak), 1)

    for event_id, (seg_start, seg_end) in enumerate(segments, start=1):
        event_mask = (ts_series >= seg_start) & (ts_series <= seg_end)
        pre_mask = (ts_series >= (seg_start - pre_window)) & (ts_series < seg_start)
        post_mask = (ts_series > seg_end) & (ts_series <= (seg_end + post_window))

        event_slice = df.loc[event_mask]
        pre_slice = df.loc[pre_mask]
        post_slice = df.loc[post_mask]

        active_metrics: list[str] = []
        for col in metric_cols:
            hit_col = f"sync_hit_{col}"
            if hit_col in event_slice.columns and event_slice[hit_col].any():
                active_metrics.append(col)

        pre_mean: dict[str, float] = {}
        peak_vals: dict[str, float] = {}
        post_mean: dict[str, float] = {}
        pre_to_peak: dict[str, float] = {}
        post_to_pre: dict[str, float] = {}
        rec_mins: dict[str, float | None] = {}

        for col in metric_cols:
            if not pre_slice.empty:
                base_val = float(pre_slice[col].mean())
                base_scale = _robust_scale(pre_slice[col], fallback_series=df[col])
            else:
                base_val = float(df[col].mean())
                base_scale = _robust_scale(df[col])

            ev_series = event_slice[col]
            peak_idx = (ev_series - base_val).abs().idxmax()
            peak_val = float(df.loc[peak_idx, col])
            after_val = float(post_slice[col].mean()) if not post_slice.empty else peak_val

            pre_mean[col] = round(base_val, 4)
            peak_vals[col] = round(peak_val, 4)
            post_mean[col] = round(after_val, 4)
            pre_to_peak[col] = round(peak_val - base_val, 4)
            post_to_pre[col] = round(after_val - base_val, 4)
            rec_mins[col] = _compute_recovery_minutes(
                post_slice=post_slice,
                col=col,
                event_end=seg_end,
                base_val=base_val,
                base_scale=base_scale,
                recovery_z_tol=recovery_z_tol,
                min_recovery_streak=streak_req,
            )

        post_samples_count = int(len(post_slice))
        duration_min = round(max((seg_end - seg_start).total_seconds() / 60.0, 1.0), 1)
        max_z = round(float(event_slice["max_metric_z"].max()), 4)
        synced_count = len(active_metrics)
        severity_tier = classify_severity_tier(
            max_composite_z=max_z,
            synced_metrics_count=synced_count,
        )

        summaries.append(
            EventBaselineSummary(
                event_id=event_id,
                start_time=seg_start.isoformat(),
                end_time=seg_end.isoformat(),
                duration_minutes=duration_min,
                synced_metrics_count=synced_count,
                synced_metrics=active_metrics,
                max_composite_z=max_z,
                severity_tier=severity_tier,
                post_samples_count=post_samples_count,
                recovery_streak_required=streak_req,
                pre_baseline_mean=pre_mean,
                peak_values=peak_vals,
                post_window_mean=post_mean,
                pre_to_peak_delta=pre_to_peak,
                post_to_pre_delta=post_to_pre,
                recovery_minutes=rec_mins,
            )
        )

    return summaries


def _format_signed(val: float) -> str:
    return f"{val:+.4f}" if val != 0 else "0.0000"


def render_markdown_report(
    summaries: list[EventBaselineSummary],
    metric_cols: list[str],
    pre_window_label: str = "2h",
    post_window_label: str = "2h",
    recovery_streak_required: int = 3,
) -> str:
    """將 EventBaselineSummary 清單渲染為 Markdown 事件前後對比報告。"""
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# Space Weather Multi-Metric Anomaly Comparison Report",
        "",
        f"- **Generated At:** {generated_at}",
        f"- **Baseline Windows:** Pre = `{pre_window_label}` before event start | Post = `{post_window_label}` after event end",
        f"- **Recovery Rule:** Requires `{recovery_streak_required}` consecutive post-event samples within the pre-window tolerance band",
        f"- **Tracked Metrics:** {', '.join(f'`{m}`' for m in metric_cols)}",
        f"- **Detected Synced Events:** {len(summaries)}",
        "- **Deterministic Severity Rules:** `CRITICAL` (z >= 6.0 & synced >= 4); `HIGH` (z >= 4.0 & synced >= 3, or z >= 6.0 & synced >= 2); `MODERATE` (z >= 3.0 & synced >= 2); otherwise `LOW`.",
        "",
    ]

    if not summaries:
        lines.extend(
            [
                "## Summary",
                "",
                "No multi-metric synchronized anomaly events met the current threshold criteria.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## 1. Event Overview",
            "",
            "| Event | Severity | UTC Window | Duration (min) | Synced Count | Synced Metrics | Max Z | Post Samples |",
            "| :--- | :--- | :--- | ---: | ---: | :--- | ---: | ---: |",
        ]
    )

    for s in summaries:
        metrics_str = ", ".join(f"`{m}`" for m in s.synced_metrics) if s.synced_metrics else "-"
        window_str = f"`{s.start_time}` ~ `{s.end_time}`"
        lines.append(
            f"| #{s.event_id} | `{s.severity_tier}` | {window_str} | {s.duration_minutes:.1f} | "
            f"{s.synced_metrics_count} | {metrics_str} | {s.max_composite_z:.4f} | {s.post_samples_count} |"
        )

    lines.append("")
    lines.append("## 2. Pre / Peak / Post Baseline Comparison by Event")
    lines.append("")

    for s in summaries:
        tail_note = (
            " *(tail cutoff: no post-window samples available after event end)*"
            if s.post_samples_count == 0
            else ""
        )
        lines.extend(
            [
                f"### Event #{s.event_id} (`{s.start_time}` -> `{s.end_time}`)",
                "",
                f"- **Severity Tier:** `{s.severity_tier}`",
                f"- **Duration:** {s.duration_minutes:.1f} min",
                f"- **Synchronized Metrics ({s.synced_metrics_count}):** {', '.join(s.synced_metrics) or 'None'}",
                f"- **Peak Composite Z-Score:** {s.max_composite_z:.4f}",
                f"- **Post-Window Samples:** {s.post_samples_count} (streak required: `{s.recovery_streak_required}`){tail_note}",
                "",
                "| Metric | Pre-Baseline Mean | Event Peak | Post-Window Mean | Pre -> Peak Delta | Post - Pre Delta | Recovery (min) |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        for col in metric_cols:
            pre_v = s.pre_baseline_mean.get(col, 0.0)
            peak_v = s.peak_values.get(col, 0.0)
            post_v = s.post_window_mean.get(col, 0.0)
            d_peak = s.pre_to_peak_delta.get(col, 0.0)
            d_post = s.post_to_pre_delta.get(col, 0.0)
            rec_v = s.recovery_minutes.get(col)
            if s.post_samples_count == 0:
                rec_str = "tail cutoff (no post data)"
            elif rec_v is not None:
                rec_str = f"{rec_v:.1f}"
            else:
                rec_str = "not recovered"
            lines.append(
                f"| `{col}` | {pre_v:.4f} | {peak_v:.4f} | {post_v:.4f} | "
                f"{_format_signed(d_peak)} | {_format_signed(d_post)} | {rec_str} |"
            )

        lines.append("")

    return "\n".join(lines)


def format_technical_summary_blocks(
    summaries: list[EventBaselineSummary],
    top_n: int = 3,
    top_metrics_per_event: int = 2,
) -> str:
    """將前 top_n 筆高偏離 EventBaselineSummary 整理為精簡技術摘要區塊。"""
    if not summaries or top_n <= 0:
        return "[No technical summary blocks generated: no qualifying events.]"

    ranked = sorted(
        summaries,
        key=lambda item: (item.max_composite_z, item.synced_metrics_count),
        reverse=True,
    )[:top_n]

    blocks: list[str] = []
    for rank, item in enumerate(ranked, start=1):
        ranked_metrics = sorted(
            item.pre_to_peak_delta.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:top_metrics_per_event]

        metric_highlights: list[str] = []
        for metric_name, peak_delta in ranked_metrics:
            post_delta = item.post_to_pre_delta.get(metric_name, 0.0)
            rec_val = item.recovery_minutes.get(metric_name)
            if item.post_samples_count == 0:
                rec_label = "tail cutoff"
            elif rec_val is not None:
                rec_label = f"{rec_val:.1f}m"
            else:
                rec_label = "unrecovered"
            metric_highlights.append(
                f"{metric_name}: peak {_format_signed(peak_delta)}, post {_format_signed(post_delta)}, rec {rec_label}"
            )

        synced_list = ", ".join(item.synced_metrics) if item.synced_metrics else "N/A"
        highlights_text = " | ".join(metric_highlights) if metric_highlights else "N/A"
        tail_suffix = " [tail cutoff: 0 post samples]" if item.post_samples_count == 0 else ""

        block = "\n".join(
            [
                f"### Brief #{rank} — Event #{item.event_id} [{item.severity_tier}] (max_z={item.max_composite_z:.2f}){tail_suffix}",
                f"- **Window (UTC):** `{item.start_time}` -> `{item.end_time}` ({item.duration_minutes:.1f} min)",
                f"- **Synced Metrics ({item.synced_metrics_count}):** {synced_list}",
                f"- **Post Samples:** {item.post_samples_count} (streak={item.recovery_streak_required})",
                f"- **Key Shifts:** {highlights_text}",
            ]
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-metric space weather anomaly detector and baseline comparison pipeline."
    )
    parser.add_argument(
        "--sync-window",
        default="30min",
        help="Rolling window for cross-metric synchronization hits (e.g., 30min, 1h). Default: 30min",
    )
    parser.add_argument(
        "--baseline-window",
        default="2h",
        help="Rolling historical window for per-metric baseline center and dispersion. Default: 2h",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.5,
        help="Normalized deviation threshold for a metric synchronization hit. Default: 2.5",
    )
    parser.add_argument(
        "--min-synced-metrics",
        type=int,
        default=2,
        help="Minimum number of synchronized metrics required for an event row. Default: 2",
    )
    parser.add_argument(
        "--pre-window",
        default="2h",
        help="Pre-event baseline window duration (e.g., 2h, 90min). Default: 2h",
    )
    parser.add_argument(
        "--post-window",
        default="2h",
        help="Post-event recovery window duration (e.g., 2h, 3h). Default: 2h",
    )
    parser.add_argument(
        "--gap-tolerance",
        default="20min",
        help="Maximum gap between flagged rows merged into one event. Default: 20min",
    )
    parser.add_argument(
        "--recovery-z-tol",
        type=float,
        default=0.75,
        help="Tolerance multiplier on pre-event scale for recovery band. Default: 0.75",
    )
    parser.add_argument(
        "--min-recovery-streak",
        type=int,
        default=3,
        help="Consecutive post-event in-band samples required to count as recovered. Default: 3",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=1500,
        help="Number of most recent merged observations to analyze. Default: 1500",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top events included in technical summary blocks. Default: 3",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    pre_window = pd.Timedelta(args.pre_window)
    post_window = pd.Timedelta(args.post_window)
    gap_tolerance = pd.Timedelta(args.gap_tolerance)
    recovery_streak_required = max(int(args.min_recovery_streak), 1)

    raw_df = fetch_merged_space_weather_df(tail_rows=args.tail_rows)
    scored_df, metric_cols = detect_synced_anomalies(
        raw_df,
        sync_window=args.sync_window,
        baseline_window=args.baseline_window,
        z_threshold=args.z_threshold,
        min_synced_metrics=args.min_synced_metrics,
    )
    summaries = build_event_baseline_summaries(
        scored_df,
        metric_cols,
        pre_window=pre_window,
        post_window=post_window,
        gap_tolerance=gap_tolerance,
        recovery_z_tol=args.recovery_z_tol,
        min_recovery_streak=recovery_streak_required,
    )

    print(f"Rows analyzed: {len(scored_df)} | Synced anomaly rows: {int(scored_df['synced_event'].sum())}")
    print(f"Consolidated multi-metric events: {len(summaries)}\n")

    scored_df.to_csv("synced_anomaly_timeseries.csv", index=False, encoding="utf-8")
    Path("event_baseline_summaries.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown_report = render_markdown_report(
        summaries=summaries,
        metric_cols=metric_cols,
        pre_window_label=args.pre_window,
        post_window_label=args.post_window,
        recovery_streak_required=recovery_streak_required,
    )
    Path("event_comparison_report.md").write_text(markdown_report, encoding="utf-8")

    technical_briefs = format_technical_summary_blocks(
        summaries,
        top_n=args.top_n,
        top_metrics_per_event=2,
    )
    Path("event_technical_briefs.md").write_text(technical_briefs + "\n", encoding="utf-8")

    print("--- Top Technical Summary Blocks ---")
    print(technical_briefs)
    print(
        "\nSaved synced_anomaly_timeseries.csv, event_baseline_summaries.json, "
        "event_comparison_report.md, and event_technical_briefs.md"
    )


if __name__ == "__main__":
    main()