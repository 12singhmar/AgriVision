"""
AgriVision - Model Evaluation
=============================
Evaluates every model on the held-out test years (2017-2020) and compares them with
two naive baselines that need no ML at all:
- "Last Year (Baseline)"    : predict this year's yield = last recorded yield
- "3-Year Avg (Baseline)"   : predict the mean of the last 3 recorded yields

Metrics: MAE, RMSE, R² and median absolute percentage error (MdAPE).
Outputs: reports/model_comparison.csv, reports/punjab_metrics.csv,
         reports/crop_errors.csv and figures in reports/figures/.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from preprocessing import from_target
from train_models import MODEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def metrics(y, p):
    nz = y > 0.05  # percentage error is undefined for ~zero yields
    return {
        "MAE": round(mean_absolute_error(y, p), 4),
        "RMSE": round(root_mean_squared_error(y, p), 4),
        "R2": round(r2_score(y, p), 4),
        "MdAPE_%": round(100 * np.median(np.abs(p[nz] - y[nz]) / y[nz]), 2),
    }


def get_predictions(X_test, test_df):
    lag1 = test_df["Yield_lag1"].values
    preds = {
        "Last Year (Baseline)": lag1,
        "3-Year Avg (Baseline)": test_df["Yield_roll3"].values,
    }
    for slug, name in MODEL_NAMES.items():
        model = joblib.load(os.path.join(MODELS_DIR, f"{slug}.joblib"))
        preds[name] = from_target(model.predict(X_test), lag1)
    return preds


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_proc.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_data.csv"))
    full_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "cleaned_crop_yield.csv"))
    preds = get_predictions(X_test, test_df)

    # 1. Overall comparison
    results = pd.DataFrame([{"Model": n, **metrics(y_test, p)} for n, p in preds.items()])
    results = results.sort_values("MAE").reset_index(drop=True)
    results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    logging.info(f"\nTEST SET ({len(y_test)} rows)\n{results.to_string(index=False)}")

    # 2. Punjab subset
    pj = (test_df["State"] == "Punjab").values
    punjab = pd.DataFrame([{"Model": n, **metrics(y_test[pj], p[pj])} for n, p in preds.items()])
    punjab = punjab.sort_values("MAE").reset_index(drop=True)
    punjab.to_csv(os.path.join(REPORTS_DIR, "punjab_metrics.csv"), index=False)
    logging.info(f"\nPUNJAB ({pj.sum()} rows)\n{punjab.to_string(index=False)}")

    # 3. Per-crop MAE: best ML model vs last-year baseline (10 most common crops)
    best = results[~results["Model"].str.contains("Baseline")].iloc[0]["Model"]
    err = test_df[["Crop"]].copy()
    err["Baseline_AE"] = np.abs(preds["Last Year (Baseline)"] - y_test)
    err["Model_AE"] = np.abs(preds[best] - y_test)
    top = err["Crop"].value_counts().head(10).index
    crop_errors = (err[err["Crop"].isin(top)].groupby("Crop")[["Baseline_AE", "Model_AE"]].mean().round(3)
                   .rename(columns={"Baseline_AE": "Last_Year_MAE", "Model_AE": f"{best}_MAE"}))
    crop_errors.to_csv(os.path.join(REPORTS_DIR, "crop_errors.csv"))
    logging.info(f"Best ML model: {best}")

    make_figures(results, preds, y_test, best, full_df)


def make_figures(results, preds, y_test, best, full_df):
    # Model comparison bars (MAE and R²)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#9e9e9e" if "Baseline" in m else "#2e7d32" for m in results["Model"]]
    for ax, col, title in [(axes[0], "MAE", "MAE (lower is better)"), (axes[1], "R2", "R² (higher is better)")]:
        ax.barh(results["Model"], results[col], color=colors)
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()
        for i, v in enumerate(results[col]):
            ax.text(v, i, f" {v:.3f}", va="center")
    plt.suptitle("Test set 2017–2020 (grey = baselines)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_bars.png"), dpi=150)
    plt.close()

    # Actual vs predicted (log axes because yields range from 0.1 to 100+ t/ha)
    p = preds[best]
    m = (y_test > 0) & (p > 0)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test[m], p[m], s=10, alpha=0.4)
    lim = [y_test[m].min(), y_test[m].max()]
    plt.plot(lim, lim, "r--", label="perfect prediction")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Actual yield (t/ha)"); plt.ylabel(f"Predicted yield – {best}")
    plt.title("Actual vs predicted (test set)", fontweight="bold")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # Residuals
    res = y_test - p
    plt.figure(figsize=(7, 4))
    plt.hist(res, bins=80, range=(-5, 5), color="#2e7d32")
    plt.title(f"Residuals (actual − predicted), {best}, zoomed to ±5 t/ha", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_distribution.png"), dpi=150)
    plt.close()

    # Punjab historical trends for the main crops
    pj = full_df[full_df["State"] == "Punjab"]
    plt.figure(figsize=(10, 5))
    for crop in ["Wheat", "Rice", "Maize", "Cotton(lint)"]:
        s = pj[pj["Crop"] == crop].groupby("Crop_Year")["Yield"].mean()
        if len(s):
            plt.plot(s.index, s.values, marker="o", label=crop)
    plt.title("Punjab: yield over time", fontweight="bold")
    plt.xlabel("Year"); plt.ylabel("Yield (t/ha)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "punjab_historical_trends.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
    print("\n[SUCCESS] Evaluation complete.")
