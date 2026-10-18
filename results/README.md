# results/ — Committed result tables

## FINAL — use these numbers

| File | What it is | Produced by |
|---|---|---|
| `multi_target_results_8models.csv` | Test MAE / RMSE / R² / MAPE, all 6 targets × 9 models | step 6 |
| `multi_target_performance_indicators.csv` | Same metrics + derived indicators (see note) | step 8 (`step8_performance_indicators.py`) |
| `multi_target_best_8models.csv` | Best model per target (lowest test MAE) | step 6 / step 8 |
| `final_best_per_target.csv` | Best per target (same as above, R version) | `check_models_accuracy_multi.R` |
| `final_overall_ranking.csv` | Overall ranking by avg MAE | `check_models_accuracy_multi.R` |

Headline: ensemble wins overall (avg MAE 0.844, avg R² 0.9214);
best per target in root README.

## EARLIER RUNS — kept for provenance, do NOT mix with final

| File | Why it differs |
|---|---|
| `multi_target_accuracy_summary.csv` | Earlier run (different ensemble `Gated+Adaptive`, different LSTM numbers, e.g. temperature LSTM MAE 0.986 vs final 0.481). Superseded. |
| `multi_target_accuracy_percentage.csv` | Different run's recomputation from a predictions file (e.g. temperature LR MAE 0.431 vs final 0.476). Superseded. |

These two are kept so the history is not hidden, but they are **not**
the final results. If you quote numbers, use the FINAL files above.

## Derived indicators

`R2_x100`, `MAPE_derived_indicator`, `MAE_derived_indicator` are
supplementary indicators, not formal accuracy — see
`src/evaluation/README.md`.
