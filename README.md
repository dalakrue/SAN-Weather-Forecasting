# SAN Weather Forecasting — Multi-Target Forecasting (San Diego, 2006–2025)

Hourly weather for San Diego (SAN), 20 years. Forecast **6 targets** with **9 models**:
temperature, relative humidity, wind speed, pressure, visibility, heat index.

Models: Linear Regression, Random Forest, LSTM, LSTM + Feature Attention,
LSTM + Temporal Attention, LSTM + Feature + Temporal Attention,
LSTM + Feature + Temporal + Adaptive, LSTM + Feature + Temporal + Adaptive + Gated Fusion,
and a **validation-weighted ensemble** of all 8.

Stack: R (data pipeline) + Python (PyTorch / scikit-learn). No web app —
this is a research pipeline with saved result tables.

## Final results (test set, 2024–2025)

Best model per target (by test MAE) — from `results/final_best_per_target.csv`:

| Target | Best model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|
| temperature_c | ENSEMBLE (Val-Weighted) | 0.458 | 0.675 | 0.9700 | 2.87% |
| relative_humidity_pct | ENSEMBLE (Val-Weighted) | 2.756 | 4.040 | 0.9168 | 4.46% |
| wind_speed_ms | LSTM + Temporal Attention* | 0.470 | 0.620 | 0.8507 | 50.60% |
| pressure_hpa | Linear Regression | 0.198 | 0.266 | 0.9940 | 0.02% |
| visibility_km | ENSEMBLE (Val-Weighted) | 0.580 | 1.226 | 0.8290 | 11.33% |
| heat_index_c | ENSEMBLE (Val-Weighted) | 0.593 | 0.835 | 0.9673 | 4.04% |

\* Wind speed: the ensemble is effectively tied (MAE 0.470, RMSE 0.619,
R² 0.8515) — see `results/final_best_per_target.csv`, which lists both.

Overall ranking (avg MAE across all 6 targets) — `results/final_overall_ranking.csv`:

1. ENSEMBLE (Val-Weighted) — 0.844 / R² 0.9214
2. LSTM — 0.858 / R² 0.9198
3. Linear Regression — 0.860 / R² 0.9190

The ensemble wins overall. Wind-speed MAPE looks bad (~51%) because
wind speed is often near zero, so percentage errors blow up even when
absolute error is small (MAE 0.47 m/s) — use MAE / R² for wind, not MAPE.

## How it works

1. **Data prep** (`src/data_prep/`) — R: merge 4 source CSVs (2006–2025),
   hourly aggregation, full hourly grid + imputation, feature engineering
   (lags, 24-h rolling stats, wind direction sin/cos, heat index, wind chill),
   PCA fit on training years only. Writes `SAN_20year_model_ready.csv`.
2. **Sequences** (`src/features/`) — 24-hour lookback windows, MinMax scaling
   fit on **train only** (≤2021). Splits: train ≤2021, val 2022–2023,
   test ≥2024.
3. **Models** (`src/models/`) — train 8 models per target (PyTorch LSTM
   variants, 30 epochs, Adam, early stopping patience 5; RF 30 trees).
   Ensemble weights from validation MAE (softmax over −MAE/std).
4. **Evaluation** (`src/evaluation/`) — test MAE / RMSE / R² / MAPE per
   target × model, best-per-target + overall ranking.
5. **Forecast plots** (`src/forecast/`) — actual vs predicted charts
   (first 300 test hours), per target and combined.

## Run it end-to-end

```bash
# 0. Get the data (large files are NOT in git — see data/README.md)
#    Place SAN_2006_2010.csv, SAN_2011_2015.csv, SAN_2016_2020.csv,
#    SAN_2021_2025.csv in data/, or run the pipeline from data/.

# 1. R data pipeline (needs R + dplyr, readr, lubridate, tidyr, zoo)
Rscript src/data_prep/pipeline_20year_san.R
# Writes SAN_20year_processed_analytics_v2.csv + SAN_20year_model_ready.csv

# 2. Python deps
pip install -r requirements.txt

# 3. Sequences (writes per-target m_X_*_<target>.npy, m_y_*, m_min_*, m_max_*)
python src/features/step5_multi.py

# 4. Train + evaluate (writes results CSVs + multi_target_preds_8models.pkl)
python src/models/step6_multi_8models.py

# 5. Report tables
python src/evaluation/step8_performance_indicators.py
python src/evaluation/step8_accuracy_percentage.py
Rscript src/evaluation/check_models_accuracy_multi.R

# 6. Forecast charts (writes forecast_*.png + multi_target_forecasts_8models.png)
python src/forecast/step7_forecast.py
```

R users can run the `.R` launchers instead of the `.py` files
(e.g. `Rscript src/features/step5_multi.R`) — they just call the
neighbouring Python file via `reticulate`.

## Project structure

| Folder | What it is | README |
|---|---|---|
| `data/` | Data sources + how to get/regenerate the large files (not in git) | [data/README.md](data/README.md) |
| `src/data_prep/` | R data pipeline (20-year hourly data → model-ready CSV) | [src/data_prep/README.md](src/data_prep/README.md) |
| `src/features/` | Sequence building + scaling (step 5) | [src/features/README.md](src/features/README.md) |
| `src/models/` | 8 models + ensemble (step 6) | [src/models/README.md](src/models/README.md) |
| `src/evaluation/` | Metrics, rankings, accuracy tables (step 8) | [src/evaluation/README.md](src/evaluation/README.md) |
| `src/forecast/` | Forecast charts (step 7) | [src/forecast/README.md](src/forecast/README.md) |
| `results/` | Committed result tables — final + earlier runs (labeled) | [results/README.md](results/README.md) |
| `docs/` | Report / deck (not in git) | [docs/README.md](docs/README.md) |

Each folder has its own README.

## Notes on this repo (what was fixed vs the shared zip)

- **Hard-coded Windows paths removed.** The four R wrappers wrote to
  `D:/Users/Lenovo LOQ/Document/R_Studio/...` and would fail on any other
  machine. They are now thin launchers that run the neighbouring `.py`
  file wherever the repo lives. The Python files are canonical — no more
  duplicated 300-line scripts inside R strings.
- **Per-target sequence files fixed.** `step5` saved one shared
  `m_X_train.npy` inside the target loop, so each target overwrote the
  previous one and only the last target's feature set survived.
  It now saves `m_X_train_<target>.npy` (etc.), and `step6` loads the
  matching per-target file. **Committed result CSVs came from the
  original run; a rerun with this fix may differ slightly.**
- **Result files labeled.** The zip contained three result sets that do
  not agree (`multi_target_results_8models.csv`,
  `multi_target_accuracy_summary.csv`,
  `multi_target_accuracy_percentage.csv`). Only the first is the final
  run — the other two are kept in `results/` clearly labeled as earlier
  runs. Use `final_*` + `multi_target_results_8models.csv` for the numbers
  in this README. See `results/README.md`.
- **Large files excluded.** Raw/source CSVs (~56 MB), the model-ready CSV
  (~32 MB), `.npy` sequences, the `.pkl` predictions, and PNG charts are
  gitignored — they are regenerated by the pipeline. See `data/README.md`
  and `.gitignore`.
- **Filename with spaces removed.** `20-year 2006-2025 weather for San Diego.R`
  is now `src/data_prep/pipeline_20year_san.R`.
