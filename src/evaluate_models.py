# AgriVision - Step 3: test every model on 2017-2020 and compare with 2 simple baselines
#
#   Last Year (Baseline)  -> this year's yield = last year's yield
#   3-Year Avg (Baseline) -> this year's yield = average of the last 3 years
#
# Metrics: MAE, RMSE, R2 and MdAPE (median % error)

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files, don't open windows
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from preprocessing import DATA_DIR, MODELS_DIR, FEATURES, from_target
from train_models import MODELS, REPORTS_DIR

FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def get_metrics(actual, pred):
    ok = actual > 0.05  # % error makes no sense when actual yield is ~0
    mdape = np.median(np.abs(pred[ok] - actual[ok]) / actual[ok]) * 100
    return {
        "MAE": round(mean_absolute_error(actual, pred), 4),
        "RMSE": round(root_mean_squared_error(actual, pred), 4),
        "R2": round(r2_score(actual, pred), 4),
        "MdAPE_%": round(mdape, 2),
    }


def compare(actual, preds):
    rows = []
    for name in preds:
        row = get_metrics(actual, preds[name])
        row["Model"] = name
        rows.append(row)
    table = pd.DataFrame(rows)[["Model", "MAE", "RMSE", "R2", "MdAPE_%"]]
    return table.sort_values("MAE").reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    test = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    full = pd.read_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    X_test = preprocessor.transform(test[FEATURES])
    actual = test["Yield"].values
    last_year = test["Yield_lag1"].values

    # predictions of the 2 baselines and the 5 models
    preds = {
        "Last Year (Baseline)": last_year,
        "3-Year Avg (Baseline)": test["Yield_roll3"].values,
    }
    for file_name, name in MODELS.items():
        model = joblib.load(os.path.join(MODELS_DIR, file_name + ".joblib"))
        z_pred = model.predict(X_test)
        preds[name] = from_target(z_pred, last_year)  # convert log change back to t/ha

    # 1. all test rows
    results = compare(actual, preds)
    results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print("\nTest set,", len(actual), "rows")
    print(results.to_string(index=False))

    # 2. only Punjab rows
    punjab = (test["State"] == "Punjab").values
    punjab_preds = {name: p[punjab] for name, p in preds.items()}
    punjab_results = compare(actual[punjab], punjab_preds)
    punjab_results.to_csv(os.path.join(REPORTS_DIR, "punjab_metrics.csv"), index=False)
    print("\nPunjab,", punjab.sum(), "rows")
    print(punjab_results.to_string(index=False))

    # 3. error per crop: best model vs last-year baseline (10 most common crops)
    best = results[~results["Model"].str.contains("Baseline")]["Model"].iloc[0]
    print("\nBest ML model:", best)
    errors = pd.DataFrame({
        "Crop": test["Crop"],
        "Last_Year_MAE": np.abs(last_year - actual),
        best + "_MAE": np.abs(preds[best] - actual),
    })
    top_crops = errors["Crop"].value_counts().head(10).index
    crop_errors = errors[errors["Crop"].isin(top_crops)].groupby("Crop").mean().round(3)
    crop_errors.to_csv(os.path.join(REPORTS_DIR, "crop_errors.csv"))

    # ---------------- figures ----------------
    # model comparison bars (grey = baseline, green = ML)
    colors = []
    for m in results["Model"]:
        colors.append("#9e9e9e" if "Baseline" in m else "#2e7d32")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.barh(results["Model"], results["MAE"], color=colors)
    ax1.set_title("MAE (lower is better)")
    ax1.invert_yaxis()
    ax2.barh(results["Model"], results["R2"], color=colors)
    ax2.set_title("R² (higher is better)")
    ax2.invert_yaxis()
    plt.suptitle("Test set 2017–2020 (grey = baselines)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_bars.png"), dpi=150)
    plt.close()

    # actual vs predicted, log axes because yields go from 0.1 to 100+ t/ha
    p = preds[best]
    keep = (actual > 0) & (p > 0)
    plt.figure(figsize=(7, 6))
    plt.scatter(actual[keep], p[keep], s=10, alpha=0.4)
    plt.plot([0.01, 300], [0.01, 300], "r--", label="perfect prediction")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual yield (t/ha)")
    plt.ylabel("Predicted yield (" + best + ")")
    plt.title("Actual vs predicted (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    # residuals
    plt.figure(figsize=(7, 4))
    plt.hist(actual - p, bins=80, range=(-5, 5), color="#2e7d32")
    plt.title("Residuals (actual - predicted), zoomed to ±5 t/ha")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_distribution.png"), dpi=150)
    plt.close()

    # Punjab yield over the years
    pj = full[full["State"] == "Punjab"]
    plt.figure(figsize=(10, 5))
    for crop in ["Wheat", "Rice", "Maize", "Cotton(lint)"]:
        yearly = pj[pj["Crop"] == crop].groupby("Crop_Year")["Yield"].mean()
        plt.plot(yearly.index, yearly.values, marker="o", label=crop)
    plt.title("Punjab: yield over time")
    plt.xlabel("Year")
    plt.ylabel("Yield (t/ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "punjab_historical_trends.png"), dpi=150)
    plt.close()
    print("Figures saved.")
