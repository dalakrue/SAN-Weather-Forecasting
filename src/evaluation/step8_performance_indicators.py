import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open("multi_target_preds_8models.pkl", "rb") as f:
    all_preds = pickle.load(f)

TARGETS = list(all_preds.keys())

# Load already-computed metrics
summary = pd.read_csv("multi_target_results_8models.csv")

# ---- Derived performance indicators (not "accuracy") ----
acc_rows = []
for _, row in summary.iterrows():
    target = row["Target"]
    d = all_preds[target]
    y_true = d["y_true"]
    mean_actual = np.mean(y_true)

    model_name = row["Model"]
    pred_key = {
        "Linear Regression": "LR",
        "Random Forest": "RF",
        "LSTM": "LSTM",
        "LSTM + Feature Attention": "Feat",
        "LSTM + Temporal Attention": "Temp",
        "LSTM + Feature + Temporal Attention": "Full",
        "LSTM + Feature + Temporal + Adaptive": "Adapt",
        "LSTM + Feature + Temporal + Adaptive + Gated Fusion": "Gated",
        "ENSEMBLE (Val-Weighted)": "Ensemble"
    }.get(model_name)
    if pred_key is None:
        continue
    pred = d[pred_key]

    mae  = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2   = r2_score(y_true, pred)
    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[mask] - pred[mask]) / y_true[mask])) * 100

    acc_rows.append({
        "Target": target, "Model": model_name,
        "MAE": round(mae, 3), "RMSE": round(rmse, 3),
        "R2": round(r2, 4), "MAPE_pct": round(mape, 2),
        "R2_x100": round(r2 * 100, 2),
        "MAPE_derived_indicator": round(100 - mape, 2),
        "MAE_derived_indicator": round((1 - mae / mean_actual) * 100, 2)
    })

acc_df = pd.DataFrame(acc_rows)
acc_df.to_csv("multi_target_performance_indicators.csv", index=False)

print("PERFORMANCE INDICATORS (MAE, RMSE, R2, MAPE)")
print("Note: R2_x100, MAPE_derived_indicator, MAE_derived_indicator are")
print("      supplementary indicators, NOT formal accuracy metrics.")
print(acc_df.to_string(index=False))

best = acc_df.loc[acc_df.groupby("Target")["MAE"].idxmin()]
print("")
print("BEST MODEL PER TARGET (by TEST MAE):")
print(best.to_string(index=False))
best.to_csv("multi_target_best_8models.csv", index=False)

print("")
print("Saved: multi_target_performance_indicators.csv")
