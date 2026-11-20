# AgriVision - Step 4: which features matter? (SHAP on XGBoost, the best model)
#
# SHAP value = how much one feature pushed one prediction up or down.
# Average of |SHAP| over many rows = how important that feature is overall.
# Note: the model predicts the CHANGE from last year, and SHAP shows
# what the model uses, not cause and effect.

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from preprocessing import DATA_DIR, MODELS_DIR, FEATURES
from train_models import REPORTS_DIR

if __name__ == "__main__":
    test = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))

    X_test = preprocessor.transform(test[FEATURES])
    feature_names = preprocessor.get_feature_names_out()

    # 500 random test rows is enough and much faster than all of them
    rng = np.random.default_rng(42)
    rows = rng.choice(len(X_test), size=500, replace=False)
    shap_values = shap.TreeExplainer(model).shap_values(X_test[rows])

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0).round(4),
    })
    importance = importance.sort_values("Mean_abs_SHAP", ascending=False).head(15)
    importance.to_csv(os.path.join(REPORTS_DIR, "shap_feature_importance.csv"), index=False)
    print(importance.to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.barh(importance["Feature"][::-1], importance["Mean_abs_SHAP"][::-1], color="#2e7d32")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 15 features - XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "figures", "shap_importance_bar.png"), dpi=150)
    plt.close()
