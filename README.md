# Space Weather Multi-Metric Anomaly Detector (MVP)

A Python prototype for ingesting multi-metric space weather time-series data, identifying synchronized anomalies across metrics, computing pre/peak/post baseline shifts and recovery durations, and exporting structured reports.

This project is an engineering prototype for statistical anomaly screening and baseline comparison. It is not an official space weather forecast or operational hazard warning system.

---

## Pipeline Overview

1. **Data Ingestion & Alignment (`fetch_merged_space_weather_df`)**
* Fetches 1-minute RTSW magnetic (`rtsw_mag_1m.json`) and solar wind (`rtsw_wind_1m.json`) JSON feeds from NOAA SWPC, parses UTC timestamps and numeric metrics (`bx_gsm`, `by_gsm`, `bz_gsm`, `bt`, `proton_speed`, `proton_density`), aligns records via `pd.merge_asof` (`2min` tolerance), and retains the trailing `tail_rows` (`1500` by default).
* Falls back to synthetic multi-metric data (`build_synthetic_multimetric_df`) if network requests fail or required columns are missing.


2. **Multi-Metric Anomaly Scoring (`detect_synced_anomalies`)**
* Computes rolling baseline center and robust scale (MAD scaled by `1.4826`, with standard deviation fallback) per metric to derive normalized deviation scores (`z_<col>`) and rolling window hit flags (`sync_hit_<col>` within `sync_window`, default `30min`, against `z_threshold`, default `2.5`).
* Fits an `IsolationForest` model (`200` estimators, `contamination=0.03`, `random_state=42`) across numeric metrics to flag multivariate outliers (`is_isolation_anomaly`) and score severity (`anomaly_severity`).
* Marks rows with at least `min_synced_metrics` (`2` by default) active window hits and an `IsolationForest` anomaly flag as `synced_event`.


3. **Event Consolidation & Baseline Analysis (`build_event_baseline_summaries`)**
* Groups consecutive `synced_event` rows within `gap_tolerance` (`20min` by default) into consolidated event segments.
* For each event, computes pre-event baseline mean and dispersion from `pre_window` (`2h` before event start), locates the peak deviation within the event window, computes post-window mean over `post_window` (`2h` after event end), calculates `pre_to_peak_delta` and `post_to_pre_delta`, records `post_samples_count`, and evaluates `recovery_minutes` and `severity_tier`.


4. **Report & Summary Generation (`render_markdown_report`, `format_technical_summary_blocks`, `main`)**
* Exports time-series scores, JSON event summaries, a full Markdown comparison report, and condensed Markdown technical briefs.



---

## Output Files

| File | Format | Contents |
| --- | --- | --- |
| `synced_anomaly_timeseries.csv` | CSV | Analyzed timestamps, raw metric values, per-metric Z-scores, rolling hit flags, `synced_metrics_count`, `max_metric_z`, `IsolationForest` flags/scores, and `synced_event`. |
| `event_baseline_summaries.json` | JSON | Serialized `EventBaselineSummary` records including window timestamps, duration, `synced_metrics`, `max_composite_z`, `severity_tier`, `post_samples_count`, `recovery_streak_required`, baseline/peak/post values, deltas, and `recovery_minutes`. |
| `event_comparison_report.md` | Markdown | Run metadata, rule summary, multi-event overview table, and per-event pre/peak/post comparison tables with tail-cutoff notes when `post_samples_count == 0`. |
| `event_technical_briefs.md` | Markdown | Condensed summary blocks for the top `N` events ranked by `max_composite_z` and `synced_metrics_count`, highlighting top metric deltas and recovery status. |

---

## Recovery and Severity Rules

### Consecutive-Streak Recovery (`_compute_recovery_minutes`)

* For each metric, `tolerance` is set to `max(recovery_z_tol * base_scale, 1e-6)` using `recovery_z_tol = 0.75` and the pre-event dispersion scale (`base_scale`).
* Post-event rows (`timestamp > seg_end` up to `seg_end + post_window`) are checked in chronological order for `abs(value - base_val) <= tolerance`.
* A metric is considered recovered only when `min_recovery_streak` (`3` by default) consecutive post-event samples fall inside the tolerance band. `recovery_minutes` is the elapsed time in minutes from `seg_end` to the timestamp of the first sample in that qualifying streak.
* If `post_samples_count == 0`, there are no post-window samples (`tail cutoff`). If post-window samples exist but do not reach the required consecutive in-band streak, `recovery_minutes` is `null` (`not recovered` / `unrecovered`).

### Deterministic Severity Tiers (`classify_severity_tier`)

Evaluated in order from `max_composite_z` and `synced_metrics_count`:

* **`CRITICAL`**: `max_composite_z >= 6.0` and `synced_metrics_count >= 4`
* **`HIGH`**: (`max_composite_z >= 4.0` and `synced_metrics_count >= 3`) or (`max_composite_z >= 6.0` and `synced_metrics_count >= 2`)
* **`MODERATE`**: `max_composite_z >= 3.0` and `synced_metrics_count >= 2`
* **`LOW`**: All other qualifying events

---

## Current Limitations

* **Statistical deviation vs. physical impact:** Z-scores, rolling scales, and `IsolationForest` detect statistical deviations relative to the observed series; they do not directly model magnetospheric coupling, propagation delay from upstream monitors, or domain hazard thresholds such as geomagnetic storm indices.
* **Directionality aggregation:** Peak shifts (`abs(ev_series - base_val).idxmax()`) and Z-scores measure magnitude rather than physical polarity (for example, southward vs. northward `bz_gsm`).
* **Sliding tail window sensitivity:** Because `fetch_merged_space_weather_df` slices the last `tail_rows` records from live feeds, re-running against advancing streams shifts the dataset window and can alter event numbering, rolling baselines, or `post_samples_count` near the end of the buffer.
* **No CLI argument parser or automated test suite yet:** Parameters such as windows, Z thresholds, and recovery streak length are currently configured in code, and automated unit tests have not yet been added to the repository.