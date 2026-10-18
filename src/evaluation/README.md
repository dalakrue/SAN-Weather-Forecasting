# src/evaluation/ — Metrics, rankings, accuracy tables (step 8)

## Files

| File | What it does |
|---|---|
| `step8_performance_indicators.py` | Recomputes MAE/RMSE/R²/MAPE from the saved predictions + adds derived indicators. Writes `multi_target_performance_indicators.csv` + `multi_target_best_8models.csv`. (Extracted from the old `step8_report.R` wrapper.) |
| `step8_report.R` | Thin launcher for `step8_performance_indicators.py`. |
| `step8_accuracy_percentage.py` | Recomputes accuracy-style percentages from the saved predictions. Writes `multi_target_accuracy_percentage.csv` (+ `multi_target_best_accuracy.csv`). (Was `step8_report_multi.py`.) |
| `check_models_accuracy_multi.R` | Best-per-target (lowest test MAE) + overall ranking by avg MAE. Writes `final_best_per_target.csv` + `final_overall_ranking.csv`. |

## How to run

```bash
# After step 6 (needs multi_target_preds_8models.pkl + result CSVs):
python src/evaluation/step8_performance_indicators.py
python src/evaluation/step8_accuracy_percentage.py
Rscript src/evaluation/check_models_accuracy_multi.R
```

## Metric honesty note

`step8_performance_indicators.py` adds `R2_x100`, `MAPE_derived_indicator`
(100 − MAPE), and `MAE_derived_indicator` (100 × (1 − MAE/mean)) as
**supplementary indicators, NOT formal accuracy metrics** — the script
itself says so. Quote MAE / RMSE / R² / MAPE as the real metrics.
Wind-speed MAPE (~51%) is misleading because wind speed is often near
zero — use MAE / R² for wind.

## Which result files are final?

See `results/README.md`. `final_*` + `multi_target_results_8models.csv` +
`multi_target_performance_indicators.csv` are the final run. The two
`multi_target_accuracy_*` files are from different runs — kept for
provenance, do not mix them with the final numbers.
