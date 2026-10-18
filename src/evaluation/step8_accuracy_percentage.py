import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open("multi_target_preds_8models.pkl", "rb") as f:
    all_preds = pickle.load(f)

TARGETS = list(all_preds.keys())

# ---------- 1. Summary table (already saved, reload) ----------
summary = pd.read_csv("multi_target_results_8models.csv")
print("========== SUMMARY METRICS (all 6 targets) ==========")
print(summary.to_string(index=False))
print("")

# ---------- 2. Accuracy in percentage ----------
acc_rows = []
for target in TARGETS:
    d = all_preds[target]
    y_true = d["y_true"]
    mean_actual = np.mean(y_true)

    for name in ["LR", "RF", "LSTM", "Feat", "Temp", "Full", "Adapt", "Gated", "Ensemble"]:
        pred = d[name]
        mae  = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        r2   = r2_score(y_true, pred)
        mask = np.abs(y_true) > 1e-6
        mape = np.mean(np.abs((y_true[mask] - pred[mask]) / y_true[mask])) * 100
        acc_rows.append({
            "Target": target, "Model": name,
            "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "R2": round(r2, 4), "MAPE_pct": round(mape, 2),
            "Acc_R2_pct": round(r2 * 100, 2),
            "Acc_MAPE_pct": round(100 - mape, 2),
            "Acc_MAE_pct": round((1 - mae / mean_actual) * 100, 2)
        })

acc_df = pd.DataFrame(acc_rows)
acc_df.to_csv("multi_target_accuracy_percentage.csv", index=False)

print("========== ACCURACY IN PERCENTAGE ==========")
print(acc_df.to_string(index=False))
print("")

# ---------- 3. Best per target ----------
best = acc_df.loc[acc_df.groupby("Target")["Acc_MAE_pct"].idxmax()]
print("========== BEST MODEL PER TARGET ==========")
print(best.to_string(index=False))
best.to_csv("multi_target_best_accuracy.csv", index=False)

print("")
print("Saved: multi_target_accuracy_percentage.csv, multi_target_best_accuracy.csv")

