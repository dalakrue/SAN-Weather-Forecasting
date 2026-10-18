import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("multi_target_preds_8models.pkl", "rb") as f:
    all_preds = pickle.load(f)

TARGETS = list(all_preds.keys())

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for i, target in enumerate(TARGETS):
    d = all_preds[target]
    ax = axes[i]
    ax.plot(d["y_true"][:300], label="Actual", color="black", linewidth=1.5)
    ax.plot(d["Ensemble"][:300], label="Ensemble (Val-Weighted)",
            color="red", linestyle="--", linewidth=2)
    ax.set_title(target + " (test 2024-2025)")
    ax.set_xlabel("Hour")
    ax.set_ylabel(target)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Model Predictions vs Actual (Test Period 2024-2025)", fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig("multi_target_forecasts_8models.png", dpi=100, bbox_inches="tight")
print("Saved: multi_target_forecasts_8models.png")

for target in TARGETS:
    d = all_preds[target]
    plt.figure(figsize=(14, 4))
    plt.plot(d["y_true"][:300], label="Actual", color="black", linewidth=1.5)
    plt.plot(d["LR"][:300], label="LR", alpha=0.5)
    plt.plot(d["RF"][:300], label="RF", alpha=0.5)
    plt.plot(d["LSTM"][:300], label="LSTM", alpha=0.5)
    plt.plot(d["Gated"][:300], label="+Gated", alpha=0.6, linewidth=1.5)
    plt.plot(d["Ensemble"][:300], label="ENSEMBLE (Val-Weighted)",
             color="red", linestyle="--", linewidth=2)
    plt.title(target + " - All 8 Models + Val-Weighted Ensemble")
    plt.xlabel("Hour")
    plt.ylabel(target)
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig("forecast_" + target + "_8models.png", dpi=100)
    print("Saved: forecast_" + target + "_8models.png")
