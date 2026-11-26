# testing all models on 2017-2020 and comparing with 2 simple baselines
# baseline 1: this year = last year
# baseline 2: this year = average of last 3 years

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import DATA_DIR, MODELS_DIR, FEATURES, from_target
from train_models import MODELS, REPORTS_DIR

FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def get_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, pred)
    # median % error, skipping rows where yield is ~0 (cant divide by 0)
    ok = actual > 0.05
    mdape = np.median(np.abs(pred[ok] - actual[ok]) / actual[ok]) * 100
    return [round(mae, 4), round(mse, 4), round(rmse, 4), round(r2, 4), round(mdape, 2)]


def make_table(actual, preds):
    rows = []
    for name in preds:
        rows.append([name] + get_metrics(actual, preds[name]))
    table = pd.DataFrame(rows, columns=["Model", "MAE", "MSE", "RMSE", "R2", "MdAPE_%"])
    table = table.sort_values("MAE").reset_index(drop=True)
    return table


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    test = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    full = pd.read_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    X_test = preprocessor.transform(test[FEATURES])
    actual = test["Yield"].values
    last_year = test["Yield_lag1"].values

    preds = {}
    preds["Last Year (Baseline)"] = last_year
    preds["3-Year Avg (Baseline)"] = test["Yield_roll3"].values
    for file_name in MODELS:
        model = joblib.load(os.path.join(MODELS_DIR, file_name + ".joblib"))
        z_pred = model.predict(X_test)
        preds[MODELS[file_name]] = from_target(z_pred, last_year)  # back to t/ha

    results = make_table(actual, preds)
    results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print("Test set,", len(actual), "rows")
    print(results.to_string(index=False))

    # only punjab rows
    pj = (test["State"] == "Punjab").values
    pj_preds = {}
    for name in preds:
        pj_preds[name] = preds[name][pj]
    punjab_results = make_table(actual[pj], pj_preds)
    punjab_results.to_csv(os.path.join(REPORTS_DIR, "punjab_metrics.csv"), index=False)
    print("\nPunjab,", pj.sum(), "rows")
    print(punjab_results.to_string(index=False))

    # best ML model = first row that is not a baseline
    ml_only = results[results["Model"].str.contains("Baseline") == False]
    best = ml_only["Model"].iloc[0]
    print("\nbest model:", best)

    # error for each crop (10 most common crops)
    errors = pd.DataFrame()
    errors["Crop"] = test["Crop"]
    errors["Last_Year_MAE"] = np.abs(last_year - actual)
    errors[best + "_MAE"] = np.abs(preds[best] - actual)
    top10 = errors["Crop"].value_counts().head(10).index
    crop_errors = errors[errors["Crop"].isin(top10)].groupby("Crop").mean().round(3)
    crop_errors.to_csv(os.path.join(REPORTS_DIR, "crop_errors.csv"))

    # ---- graphs ----

    # bar chart, grey = baselines, green = models
    colors = []
    for m in results["Model"]:
        if "Baseline" in m:
            colors.append("grey")
        else:
            colors.append("green")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.barh(results["Model"], results["MAE"], color=colors)
    ax1.set_title("MAE (lower is better)")
    ax1.invert_yaxis()
    ax2.barh(results["Model"], results["R2"], color=colors)
    ax2.set_title("R2 (higher is better)")
    ax2.invert_yaxis()
    plt.suptitle("Test set 2017-2020 (grey = baselines)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_bars.png"), dpi=150)
    plt.close()

    # actual vs predicted (log scale since yield goes from 0.1 to 100+)
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

    # errors histogram
    plt.figure(figsize=(7, 4))
    plt.hist(actual - p, bins=80, range=(-5, 5), color="green")
    plt.title("Residuals (actual - predicted), between -5 and 5 t/ha")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_distribution.png"), dpi=150)
    plt.close()

    # punjab yield over the years
    punjab = full[full["State"] == "Punjab"]
    plt.figure(figsize=(10, 5))
    for crop in ["Wheat", "Rice", "Maize", "Cotton(lint)"]:
        yearly = punjab[punjab["Crop"] == crop].groupby("Crop_Year")["Yield"].mean()
        plt.plot(yearly.index, yearly.values, marker="o", label=crop)
    plt.title("Punjab: yield over time")
    plt.xlabel("Year")
    plt.ylabel("Yield (t/ha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "punjab_historical_trends.png"), dpi=150)
    plt.close()
    print("graphs saved")
