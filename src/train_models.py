"""
AgriVision - Model Training
===========================
Trains 5 regression models on the log-change target (see preprocessing.to_target):
1. Linear Regression   - simple linear baseline
2. Ridge Regression    - linear + L2 penalty on coefficients
3. Random Forest       - bagging: many independent trees, averaged
4. Gradient Boosting   - boosting: trees added one by one to fix previous errors
5. XGBoost             - regularised, faster gradient boosting

Random Forest and XGBoost are tuned with RandomizedSearchCV using TimeSeriesSplit
(the training data is sorted by year, so each fold validates on later years).
"""

import os
import json
import logging
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
N_JOBS = 2  # -1 ran out of resources on Windows

MODEL_NAMES = {
    "linear_regression": "Linear Regression",
    "ridge_regression": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def tune(estimator, param_dist, X, z, name):
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=8, cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=N_JOBS,
    )
    search.fit(X, z)
    logging.info(f"Best {name} params: {search.best_params_}")
    return search.best_params_


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    X = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train_proc.npy"))
    z = np.load(os.path.join(PROCESSED_DATA_DIR, "z_train.npy"))
    logging.info(f"Training on X={X.shape}")

    rf_params = tune(
        RandomForestRegressor(random_state=42, n_jobs=N_JOBS),
        {"n_estimators": [200, 300], "min_samples_leaf": [1, 3, 5, 10],
         "max_features": [0.3, 0.5, 0.8], "max_depth": [10, 20, None]},
        X, z, "Random Forest",
    )
    xgb_params = tune(
        XGBRegressor(random_state=42, n_jobs=N_JOBS),
        {"n_estimators": [200, 400, 600], "learning_rate": [0.03, 0.05, 0.1],
         "max_depth": [3, 5, 7], "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0]},
        X, z, "XGBoost",
    )
    with open(os.path.join(REPORTS_DIR, "hyperparameter_tuning.json"), "w") as f:
        json.dump({"Random Forest": rf_params, "XGBoost": xgb_params}, f, indent=4)

    models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(**rf_params, random_state=42, n_jobs=N_JOBS),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                                       subsample=0.8, loss="huber", random_state=42),
        "xgboost": XGBRegressor(**xgb_params, random_state=42, n_jobs=N_JOBS),
    }
    for slug, model in models.items():
        model.fit(X, z)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{slug}.joblib"))
        logging.info(f"Saved {MODEL_NAMES[slug]}")


if __name__ == "__main__":
    main()
    print("\n[SUCCESS] All models trained.")
