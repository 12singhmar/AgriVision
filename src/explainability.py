# SHAP on xgboost to see which features matter the most
# note: shap shows what the model uses, not cause and effect

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

    # using 500 random rows, all rows takes too long
    rng = np.random.default_rng(42)
    rows = rng.choice(len(X_test), size=500, replace=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[rows])

    # importance = average of |shap value| for each feature
    importance = pd.DataFrame()
    importance["Feature"] = feature_names
    importance["Mean_abs_SHAP"] = np.abs(shap_values).mean(axis=0).round(4)
    importance = importance.sort_values("Mean_abs_SHAP", ascending=False).head(15)
    importance.to_csv(os.path.join(REPORTS_DIR, "shap_feature_importance.csv"), index=False)
    print(importance)

    plt.figure(figsize=(8, 6))
    plt.barh(importance["Feature"][::-1], importance["Mean_abs_SHAP"][::-1], color="green")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 15 features - XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "figures", "shap_importance_bar.png"), dpi=150)
    plt.close()
