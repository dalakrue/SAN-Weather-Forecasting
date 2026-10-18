# src/forecast/ — Forecast charts (step 7)

## Files

| File | What it is |
|---|---|
| `step7_forecast.py` | Canonical code. Plots actual vs predicted. |
| `step7_forecast.R` | Thin launcher — runs the neighbouring `.py` via `reticulate`. |

## How to run

```bash
# After step 6 (needs multi_target_preds_8models.pkl):
python src/forecast/step7_forecast.py
# or: Rscript src/forecast/step7_forecast.R
# Writes: multi_target_forecasts_8models.png (3×2 grid, all targets)
#         forecast_<target>_8models.png (one per target)
```

Python deps: `pandas`, `numpy`, `matplotlib`.

## What it plots

First 300 test hours (2024–2025), actual (black) vs ensemble
(red dashed) in the combined chart; per-target charts also show
LR, RF, LSTM, and Gated. Charts are gitignored — regenerate them
with this script.
