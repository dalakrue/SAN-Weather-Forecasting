# Step 5 — Build 24-hour sequences + MinMax scaling (fit on train only).
# Run from the repo root (or any dir containing SAN_20year_model_ready.csv).
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("SAN_20year_model_ready.csv")

TARGETS = ["temperature_c", "relative_humidity_pct",
           "wind_speed_ms", "pressure_hpa",
           "visibility_km", "heat_index_c"]

train_mask = df["year"] <= 2021
val_mask   = (df["year"] >= 2022) & (df["year"] <= 2023)
test_mask  = df["year"] >= 2024

print("Train rows:", train_mask.sum())
print("Val rows:  ", val_mask.sum())
print("Test rows: ", test_mask.sum())

feature_cols = [c for c in df.columns if c not in TARGETS + ["year"]]

# ---- Fit MinMax on TRAINING ROWS ONLY ----
scaler = MinMaxScaler()
scaler.fit(df.loc[train_mask, feature_cols + TARGETS])

df_scaled = df.copy()
df_scaled[feature_cols + TARGETS] = scaler.transform(df[feature_cols + TARGETS])

def create_sequences(features, target, years, lookback=24):
    X, y, ys = [], [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i+lookback])
        y.append(target[i+lookback])
        ys.append(years[i+lookback])
    return np.array(X), np.array(y), np.array(ys)

LOOKBACK = 24

for target in TARGETS:
    feat_cols = [c for c in feature_cols if c != target]
    X_all = df_scaled[feat_cols].values
    y_all = df_scaled[target].values
    years_all = df_scaled["year"].values

    X_seq, y_seq, y_years = create_sequences(X_all, y_all, years_all, LOOKBACK)

    train_idx = y_years <= 2021
    val_idx   = (y_years >= 2022) & (y_years <= 2023)
    test_idx  = y_years >= 2024

    # FIX (repo cleanup): save per-target X files. The original saved a
    # shared m_X_train.npy / m_X_val.npy / m_X_test.npy inside the loop,
    # so each target overwrote the previous one and only the last
    # target's (heat_index_c) feature set survived. Step6 now loads
    # the matching per-target file. Committed result CSVs came from
    # the original run; a rerun with this fix may differ slightly.
    np.save(f"m_X_train_{target}.npy", X_seq[train_idx])
    np.save(f"m_y_train_{target}.npy", y_seq[train_idx])
    np.save(f"m_X_val_{target}.npy", X_seq[val_idx])
    np.save(f"m_y_val_{target}.npy", y_seq[val_idx])
    np.save(f"m_X_test_{target}.npy", X_seq[test_idx])
    np.save(f"m_y_test_{target}.npy", y_seq[test_idx])

    col_idx = (feature_cols + TARGETS).index(target)
    tmin = scaler.data_min_[col_idx]
    tmax = scaler.data_max_[col_idx]
    np.save(f"m_min_{target}.npy", tmin)
    np.save(f"m_max_{target}.npy", tmax)

    print(f"{target:28s} train={X_seq[train_idx].shape} val={X_seq[val_idx].shape} test={X_seq[test_idx].shape}")

print("")
print("Done.")

