# src/models/ — 8 models + val-weighted ensemble (step 6)

## Files

| File | What it is |
|---|---|
| `step6_multi_8models.py` | Canonical code. Trains + evaluates all models. |
| `step6_multi_8models.R` | Thin launcher — runs the neighbouring `.py` via `reticulate`. |

## How to run

```bash
# After step 5 (needs per-target m_X_* / m_y_* / m_min_* / m_max_*.npy):
python src/models/step6_multi_8models.py
# or: Rscript src/models/step6_multi_8models.R
# Writes: multi_target_results_8models.csv, multi_target_best_8models.csv,
#         multi_target_preds_8models.pkl
```

Python deps: `torch`, `scikit-learn`, `pandas`, `numpy`.
Uses CUDA if available, else CPU. Seeds: `torch 42`, `numpy 42`.

## Models (per target)

| # | Model | Key settings |
|---|---|---|
| 1 | Linear Regression | `sklearn`, on flattened sequences |
| 2 | Random Forest | 30 trees, max_depth 15, `random_state=42` |
| 3 | LSTM | 1 layer, hidden 48, Adam lr 5e-4, 30 epochs, early stop patience 5 |
| 4 | LSTM + Feature Attention | Sigmoid feature gate before LSTM |
| 5 | LSTM + Temporal Attention | Softmax time weights over LSTM outputs |
| 6 | LSTM + Feature + Temporal Attention | Both attentions |
| 7 | + Adaptive | Adaptive scaling/shift on the context vector |
| 8 | + Gated Fusion | Gate fusing adaptive context + last hidden state |
| 9 | ENSEMBLE (Val-Weighted) | Softmax weights from validation MAE |

## Ensemble

Weights `w = softmax(−val_MAE / std(val_MAEs))` per target, computed on
validation predictions. Test prediction = weighted sum of the 8 models'
inverse-scaled test predictions.

## Metrics

Per target × model: MAE, RMSE, R², MAPE on the test set (inverse-scaled
to original units). See `results/` for the committed tables.
