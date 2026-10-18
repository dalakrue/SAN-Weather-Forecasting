# src/features/ — Sequences + scaling (step 5)

## Files

| File | What it is |
|---|---|
| `step5_multi.py` | Canonical code. Builds 24-h sequences + MinMax scaling. |
| `step5_multi.R` | Thin launcher — runs the neighbouring `.py` via `reticulate`. |

## How to run

```bash
# From a dir containing SAN_20year_model_ready.csv:
python src/features/step5_multi.py
# or: Rscript src/features/step5_multi.R
```

Python deps: `pandas`, `numpy`, `scikit-learn` (see root `requirements.txt`).

## What it does

- Splits by year: train ≤2021, val 2022–2023, test ≥2024.
- Fits `MinMaxScaler` on **training rows only**, transforms all rows.
- 24-hour lookback windows: `X = features[i:i+24]`, `y = target[i+24]`.
- Per target, excludes that target's own column from its feature set.
- Writes per-target files:
  `m_X_train_<target>.npy`, `m_X_val_<target>.npy`,
  `m_X_test_<target>.npy`, `m_y_train_<target>.npy`,
  `m_y_val_<target>.npy`, `m_y_test_<target>.npy`,
  `m_min_<target>.npy`, `m_max_<target>.npy`.

## Fix vs the original

The original saved one shared `m_X_train.npy` inside the target loop,
so each target overwrote the previous one. This version saves per-target
files, and `step6` loads the matching one. Committed result CSVs came
from the original run — a rerun may differ slightly.
