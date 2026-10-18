# Step 6 — Train 8 models + val-weighted ensemble per target.
# Requires step5 outputs (per-target m_X_*_<target>.npy, m_y_*, m_min_*, m_max_*).
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

TARGETS = ["temperature_c", "relative_humidity_pct",
           "wind_speed_ms", "pressure_hpa",
           "visibility_km", "heat_index_c"]

def make_loader(X, y, bs=512, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def train_torch(model, Xtr, ytr, Xva, yva, epochs=30, lr=5e-4, patience=5):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    tr = make_loader(Xtr, ytr)
    va = make_loader(Xva, yva, shuffle=False)

    best_val = float("inf")
    best_state = None
    pc = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                vl += loss_fn(model(xb), yb).item()
        vl /= len(va)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1

        if (ep+1) % 5 == 0:
            print("    Epoch " + str(ep+1) + "/" + str(epochs)
                  + " val=" + str(round(vl, 5))
                  + " best=" + str(round(best_val, 5)))

        if pc >= patience:
            print("    Early stop at epoch " + str(ep+1))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_torch(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.FloatTensor(X[i:i+1024]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds).flatten()

class LSTMModel(nn.Module):
    def __init__(self, nf, h=48):
        super().__init__()
        self.lstm = nn.LSTM(nf, h, 1, batch_first=True)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class FeatureOnlyAttention(nn.Module):
    def __init__(self, nf, h=48):
        super().__init__()
        self.feat_attn = nn.Sequential(nn.Linear(nf, nf), nn.Sigmoid())
        self.lstm = nn.LSTM(nf, h, 1, batch_first=True)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        x = x * self.feat_attn(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TemporalOnlyAttention(nn.Module):
    def __init__(self, nf, h=48):
        super().__init__()
        self.lstm = nn.LSTM(nf, h, 1, batch_first=True)
        self.temp_attn = nn.Linear(h, 1)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.temp_attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.fc(ctx)

class FullAttention(nn.Module):
    def __init__(self, nf, h=48):
        super().__init__()
        self.feat_attn = nn.Sequential(nn.Linear(nf, nf), nn.Sigmoid())
        self.lstm = nn.LSTM(nf, h, 1, batch_first=True)
        self.temp_attn = nn.Linear(h, 1)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        x = x * self.feat_attn(x)
        out, _ = self.lstm(x)
        w = torch.softmax(self.temp_attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.fc(ctx)

class AdaptiveScaling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scale = nn.Linear(hidden_dim, hidden_dim)
        self.shift = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        return self.scale(x) * x + self.shift(x)

class FullAttentionAdaptive(nn.Module):
    def __init__(self, nf, h=48):
        super().__init__()
        self.feat_attn = nn.Sequential(nn.Linear(nf, nf), nn.Sigmoid())
        self.lstm = nn.LSTM(nf, h, 1, batch_first=True)
        self.temp_attn = nn.Linear(h, 1)
        self.adaptive = AdaptiveScaling(h)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        x = x * self.feat_attn(x)
        out, _ = self.lstm(x)
        w = torch.softmax(self.temp_attn(out), dim=1)
        ctx = (w * out).sum(dim=1)
        ctx = self.adaptive(ctx)
        return self.fc(ctx)

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
    def forward(self, h1, h2):
        g = self.gate(torch.cat([h1, h2], dim=-1))
        return g * h1 + (1 - g) * h2

class FullAttentionAdaptiveGated(nn.Module):
    def __init__(self, nf, h=48):
        super().__init__()
        self.feat_attn = nn.Sequential(nn.Linear(nf, nf), nn.Sigmoid())
        self.lstm = nn.LSTM(nf, h, 1, batch_first=True)
        self.temp_attn = nn.Linear(h, 1)
        self.adaptive = AdaptiveScaling(h)
        self.gated = GatedFusion(h)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        x = x * self.feat_attn(x)
        out, _ = self.lstm(x)
        w = torch.softmax(self.temp_attn(out), dim=1)
        ctx_temporal = (w * out).sum(dim=1)
        ctx_adaptive = self.adaptive(ctx_temporal)
        h_feat = out[:, -1, :]
        fused = self.gated(ctx_adaptive, h_feat)
        return self.fc(fused)

# FIX (repo cleanup): load the per-target X files written by step5
# (m_X_train_<target>.npy, ...). The original loaded one shared
# m_X_train.npy for every target, which was the last target's
# feature set. Committed result CSVs came from the original run;
# a rerun with this fix may differ slightly.
all_results = []
all_predictions = {}

for target in TARGETS:
    print("Loading sequences for", target, "...")
    X_train = np.load("m_X_train_" + target + ".npy")
    X_val   = np.load("m_X_val_" + target + ".npy")
    X_test  = np.load("m_X_test_" + target + ".npy")
    n_features = X_train.shape[2]

    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)
    print("")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)
    print("========================================")
    print("TARGET:", target)
    print("========================================")
    t0 = time.time()

    y_train = np.load("m_y_train_" + target + ".npy")
    y_val   = np.load("m_y_val_" + target + ".npy")
    y_test  = np.load("m_y_test_" + target + ".npy")

    smin = float(np.load("m_min_" + target + ".npy"))
    smax = float(np.load("m_max_" + target + ".npy"))
    def inv(y):
        return y * (smax - smin) + smin

    y_true_test = inv(y_test)
    y_true_val  = inv(y_val)

    print("  1/8 Linear Regression...")
    lr = LinearRegression().fit(X_train_flat, y_train)
    lr_val  = lr.predict(X_val_flat)
    lr_test = lr.predict(X_test_flat)

    print("  2/8 Random Forest...")
    rf = RandomForestRegressor(n_estimators=30, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train_flat, y_train)
    rf_val  = rf.predict(X_val_flat)
    rf_test = rf.predict(X_test_flat)

    print("  3/8 LSTM...")
    lstm = train_torch(LSTMModel(n_features), X_train, y_train, X_val, y_val)
    lstm_val  = predict_torch(lstm, X_val)
    lstm_test = predict_torch(lstm, X_test)
    torch.cuda.empty_cache()

    print("  4/8 LSTM + Feature Attention...")
    feat = train_torch(FeatureOnlyAttention(n_features), X_train, y_train, X_val, y_val)
    feat_val  = predict_torch(feat, X_val)
    feat_test = predict_torch(feat, X_test)
    torch.cuda.empty_cache()

    print("  5/8 LSTM + Temporal Attention...")
    temp = train_torch(TemporalOnlyAttention(n_features), X_train, y_train, X_val, y_val)
    temp_val  = predict_torch(temp, X_val)
    temp_test = predict_torch(temp, X_test)
    torch.cuda.empty_cache()

    print("  6/8 LSTM + Feature + Temporal...")
    full = train_torch(FullAttention(n_features), X_train, y_train, X_val, y_val)
    full_val  = predict_torch(full, X_val)
    full_test = predict_torch(full, X_test)
    torch.cuda.empty_cache()

    print("  7/8 LSTM + Feature + Temporal + Adaptive...")
    adap = train_torch(FullAttentionAdaptive(n_features), X_train, y_train, X_val, y_val)
    adap_val  = predict_torch(adap, X_val)
    adap_test = predict_torch(adap, X_test)
    torch.cuda.empty_cache()

    print("  8/8 LSTM + Feature + Temporal + Adaptive + Gated...")
    gated = train_torch(FullAttentionAdaptiveGated(n_features), X_train, y_train, X_val, y_val)
    gated_val  = predict_torch(gated, X_val)
    gated_test = predict_torch(gated, X_test)
    torch.cuda.empty_cache()

    # ---- Ensemble weights from VALIDATION MAE ----
    val_preds = {
        "Linear Regression": lr_val,
        "Random Forest": rf_val,
        "LSTM": lstm_val,
        "LSTM + Feature Attention": feat_val,
        "LSTM + Temporal Attention": temp_val,
        "LSTM + Feature + Temporal Attention": full_val,
        "LSTM + Feature + Temporal + Adaptive": adap_val,
        "LSTM + Feature + Temporal + Adaptive + Gated Fusion": gated_val
    }
    test_preds = {
        "Linear Regression": lr_test,
        "Random Forest": rf_test,
        "LSTM": lstm_test,
        "LSTM + Feature Attention": feat_test,
        "LSTM + Temporal Attention": temp_test,
        "LSTM + Feature + Temporal Attention": full_test,
        "LSTM + Feature + Temporal + Adaptive": adap_test,
        "LSTM + Feature + Temporal + Adaptive + Gated Fusion": gated_test
    }

    val_maes = {k: mean_absolute_error(y_true_val, inv(v)) for k, v in val_preds.items()}
    std_mae = np.std(list(val_maes.values())) + 1e-8
    w = np.exp(-np.array(list(val_maes.values())) / std_mae)
    w = w / w.sum()
    weights_dict = dict(zip(val_maes.keys(), w))

    print("  Validation MAE per model:")
    for k, m in val_maes.items():
        print("    " + k.ljust(58) + " MAE=" + str(round(m, 4)) + "  w=" + str(round(weights_dict[k], 4)))

    ensemble_test = sum(weights_dict[k] * inv(test_preds[k]) for k in test_preds)

    for name, pred in list(test_preds.items()) + [("ENSEMBLE (Val-Weighted)", None)]:
        if name == "ENSEMBLE (Val-Weighted)":
            pred_c = ensemble_test
        else:
            pred_c = inv(pred)

        mae  = mean_absolute_error(y_true_test, pred_c)
        rmse = np.sqrt(mean_squared_error(y_true_test, pred_c))
        r2   = r2_score(y_true_test, pred_c)
        mask = np.abs(y_true_test) > 1e-6
        mape = np.mean(np.abs((y_true_test[mask] - pred_c[mask]) / y_true_test[mask])) * 100
        all_results.append({
            "Target": target, "Model": name,
            "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "R2": round(r2, 4), "MAPE_pct": round(mape, 2)
        })

    all_predictions[target] = {
        "y_true": y_true_test,
        "LR": inv(lr_test), "RF": inv(rf_test),
        "LSTM": inv(lstm_test), "Feat": inv(feat_test), "Temp": inv(temp_test),
        "Full": inv(full_test), "Adapt": inv(adap_test), "Gated": inv(gated_test),
        "Ensemble": ensemble_test,
        "Val_Weights": weights_dict
    }

    print("  Finished " + target + " in " + str(round((time.time() - t0) / 60, 1)) + " min")
    print("")

summary = pd.DataFrame(all_results)
summary.to_csv("multi_target_results_8models.csv", index=False)

print("ALL RESULTS")
print(summary.to_string(index=False))

best = summary.loc[summary.groupby("Target")["MAE"].idxmin()]
print("")
print("BEST PER TARGET:")
print(best.to_string(index=False))
best.to_csv("multi_target_best_8models.csv", index=False)

with open("multi_target_preds_8models.pkl", "wb") as f:
    pickle.dump(all_predictions, f)

print("")
print("Saved: multi_target_results_8models.csv, multi_target_best_8models.csv")

