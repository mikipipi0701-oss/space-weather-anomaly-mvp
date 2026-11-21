```markdown
# Space Weather Multi-Metric Anomaly Detector (MVP)

A reproducible Python prototype for ingesting multi-metric space weather time-series data, detecting synchronized anomalies across metrics, computing pre/peak/post baseline shifts and consecutive-streak recovery durations, and exporting structured CSV, JSON, and Markdown reports.

> **Scope & Disclaimer:** This project is an engineering prototype for statistical anomaly screening and baseline comparison. It is **not** an official space weather forecast or operational hazard warning system.

---

## Project Structure

```text
space-anomaly-mvp/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   └── detect_space_anomalies.py
└── tests/
    └── test_detect_space_anomalies.py

```

---

## Quickstart & CLI Usage

### 1. Environment Setup and Unit Tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -v

```

### 2. Run the Pipeline

Run with default parameters:

```powershell
python src/detect_space_anomalies.py

```

Run with custom CLI options:

```powershell
python src/detect_space_anomalies.py `
  --sync-window 30min `
  --baseline-window 2h `
  --z-threshold 2.5 `
  --min-synced-metrics 2 `
  --pre-window 2h `
  --post-window 2h `
  --gap-tolerance 20min `
  --recovery-z-tol 0.75 `
  --min-recovery-streak 3 `
  --tail-rows 1500 `
  --top-n 3

```

---

## Pipeline Steps

1. **Data Ingestion & Alignment (`fetch_merged_space_weather_df`)**
* Fetches 1-minute RTSW magnetic (`rtsw_mag_1m.json`) and solar wind (`rtsw_wind_1m.json`) JSON feeds from NOAA SWPC, parses UTC timestamps and numeric metrics (`bx_gsm`, `by_gsm`, `bz_gsm`, `bt`, `proton_speed`, `proton_density`), aligns records with `pd.merge_asof` (`2min` tolerance), and slices the trailing `tail_rows` records.
* Falls back to synthetic multi-metric data (`build_synthetic_multimetric_df`) when network requests fail or required columns are missing.


2. **Multi-Metric Anomaly Scoring (`detect_synced_anomalies`)**
* Computes rolling baseline center and robust dispersion (MAD scaled by `1.4826`, with standard deviation fallback) per metric to derive normalized deviation scores (`z_<col>`) and rolling synchronization hit flags (`sync_hit_<col>`) within `sync_window` against `z_threshold`.
* Fits an `IsolationForest` model (`200` estimators, `contamination=0.03`, `random_state=42`) across numeric metrics to flag multivariate outliers (`is_isolation_anomaly`) and compute severity scores (`anomaly_severity`).
* Flags rows with `synced_metrics_count >= min_synced_metrics` and `is_isolation_anomaly == True` as `synced_event`.


3. **Event Consolidation & Baseline Analysis (`build_event_baseline_summaries`)**
* Consolidates `synced_event` rows within `gap_tolerance` into event segments.
* Computes pre-event baseline mean and scale over `pre_window`, extracts peak deviation values within each event, computes post-window mean over `post_window`, calculates `pre_to_peak_delta` and `post_to_pre_delta`, tracks `post_samples_count`, and evaluates `recovery_minutes` and `severity_tier`.


4. **Report Export (`render_markdown_report`, `format_technical_summary_blocks`, `main`)**
* Writes time-series scores, JSON event summaries, a full Markdown comparison report, and condensed Markdown technical briefs.



---

## Output Files

| File | Format | Contents |
| --- | --- | --- |
| `synced_anomaly_timeseries.csv` | CSV | Timestamps, raw metric values, per-metric Z-scores, rolling hit flags, `synced_metrics_count`, `max_metric_z`, `IsolationForest` flags/scores, and `synced_event`. |
| `event_baseline_summaries.json` | JSON | Serialized `EventBaselineSummary` records with timestamps, duration, `synced_metrics`, `max_composite_z`, `severity_tier`, `post_samples_count`, `recovery_streak_required`, baseline/peak/post values, deltas, and `recovery_minutes`. |
| `event_comparison_report.md` | Markdown | Run metadata, rule summary, overview table, and per-event pre/peak/post comparison tables with explicit tail-cutoff notes when `post_samples_count == 0`. |
| `event_technical_briefs.md` | Markdown | Condensed briefs for the top `N` events ranked by `max_composite_z` and `synced_metrics_count`. |

---

## Recovery, Tail Cutoff, and Severity Rules

### Consecutive-Streak Recovery (`_compute_recovery_minutes`)

* Sets per-metric tolerance to `max(recovery_z_tol * base_scale, 1e-6)` using `recovery_z_tol` and pre-event scale (`base_scale`).
* Evaluates post-event samples (`timestamp > seg_end` up to `seg_end + post_window`) in chronological order for `abs(value - base_val) <= tolerance`.
* Requires `min_recovery_streak` consecutive in-band samples; resets the streak counter whenever a sample falls outside the band.
* Returns elapsed minutes from `seg_end` to the **first** timestamp of the qualifying streak, or `None` if `post_samples_count == 0` (tail cutoff) or no qualifying streak occurs within `post_window`.

### Deterministic Severity Tiers (`classify_severity_tier`)

Evaluated in priority order from `max_composite_z` and `synced_metrics_count`:

* **`CRITICAL`**: `max_composite_z >= 6.0` and `synced_metrics_count >= 4`
* **`HIGH`**: (`max_composite_z >= 4.0` and `synced_metrics_count >= 3`) or (`max_composite_z >= 6.0` and `synced_metrics_count >= 2`)
* **`MODERATE`**: `max_composite_z >= 3.0` and `synced_metrics_count >= 2`
* **`LOW`**: All other qualifying events

---

## Example Event Profiles

| Profile | Severity | Duration & Sync | Post Samples | Key Observations |
| --- | --- | --- | --- | --- |
| **Multi-Metric Surge** | `CRITICAL` | `52.0 min` (4 metrics: `bx_gsm`, `bz_gsm`, `bt`, `proton_speed`) | `284` | `proton_speed` peak delta `+27.5400`, post-pre delta `+3.7447`, recovery `1.0m`; `bz_gsm` peak delta `+5.5841`, recovery `0.0m`. |
| **Tail-Cutoff Event** | `MODERATE` | `15.0 min` (3 metrics: `bx_gsm`, `bz_gsm`, `proton_speed`) | `0` | Marked as tail cutoff (`0 post samples`); recovery is `None` until subsequent samples arrive. |
| **Brief Low-Tier Spike** | `LOW` | `1.0 min` (2 metrics: `by_gsm`, `proton_speed`) | `360` | Short-lived shift with `proton_speed` recovery `1.0m` and `by_gsm` recovery `4.0m`. |

---

## Current Limitations

* **Statistical screening vs. physical hazard models:** Rolling robust Z-scores and `IsolationForest` detect statistical anomalies; they do not model magnetospheric coupling, propagation delay, or official geomagnetic storm thresholds.
* **Magnitude vs. physical polarity:** Peak selection (`(ev_series - base_val).abs().idxmax()`) and Z-scores evaluate deviation magnitude rather than directional polarity such as southward vs. northward `bz_gsm`.
* **Sliding tail buffer behavior:** Slicing the last `tail_rows` records shifts the observation window on subsequent runs, which can change event numbering, rolling baselines, or `post_samples_count` near the tail.

```
