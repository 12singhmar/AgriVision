"""
AgriVision - Feature importance with SHAP
=========================================
Uses SHAP TreeExplainer on the best tree model (lowest test MAE among Random Forest,
Gradient Boosting and XGBoost) to rank which inputs move its predictions the most.

Remember: the models predict the CHANGE in yield versus last year, so SHAP here answers
"which inputs push this year's yield above or below last year's?".
SHAP shows association inside the model, not cause and effect in the field.

Outputs: reports/figures/shap_importance_bar.png, reports/shap_feature_importance.csv
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

TREE_MODELS = {"Random Forest": "random_forest", "Gradient Boosting": "gradient_boosting", "XGBoost": "xgboost"}


def main(sample_size: int = 500):
    with open(os.path.join(PROCESSED_DATA_DIR, "metadata.json")) as f:
        feature_names = json.load(f)["all_feature_names"]
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_proc.npy"))

    results = pd.read_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))
    best = results[results["Model"].isin(TREE_MODELS)].iloc[0]["Model"]
    model = joblib.load(os.path.join(MODELS_DIR, f"{TREE_MODELS[best]}.joblib"))
    logging.info(f"Explaining {best}")

    rng = np.random.default_rng(42)
    X_sample = X_test[rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)]
    shap_values = shap.TreeExplainer(model).shap_values(X_sample)

    importance = pd.DataFrame({"Feature": feature_names, "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0)})
    importance = importance.sort_values("Mean_abs_SHAP", ascending=False).head(15).reset_index(drop=True)
    importance["Mean_abs_SHAP"] = importance["Mean_abs_SHAP"].round(4)
    importance.to_csv(os.path.join(REPORTS_DIR, "shap_feature_importance.csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(importance["Feature"][::-1], importance["Mean_abs_SHAP"][::-1], color="#2e7d32")
    plt.xlabel("Mean |SHAP value| (effect on predicted log-change)")
    plt.title(f"Top 15 features – {best}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_bar.png"), dpi=150)
    plt.close()
    logging.info(f"\n{importance.to_string(index=False)}")


if __name__ == "__main__":
    main()
    print("\n[SUCCESS] SHAP analysis complete.")
