# AgriVision - Step 2: train 5 models
#
#   Linear Regression  - straight-line fit
#   Ridge Regression   - linear + L2 penalty (keeps coefficients small)
#   Random Forest      - bagging: many trees on random samples, averaged
#   Gradient Boosting  - boosting: each new tree fixes the errors of the previous ones
#   XGBoost            - faster boosting with regularisation
#
# All models learn the log change from last year (see to_target in preprocessing.py).

import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from preprocessing import DATA_DIR, MODELS_DIR, BASE_DIR, FEATURES, to_target

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
N_JOBS = 2  # -1 crashed on my Windows laptop (out of resources)

# file name -> display name (also used by evaluate_models.py and the app)
MODELS = {
    "linear_regression": "Linear Regression",
    "ridge_regression": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def tune(model, params, X, z):
    # TimeSeriesSplit: the data is sorted by year, so each fold validates on LATER years
    search = RandomizedSearchCV(model, params, n_iter=8, cv=TimeSeriesSplit(n_splits=3),
                                scoring="neg_root_mean_squared_error", random_state=42, n_jobs=N_JOBS)
    search.fit(X, z)
    return search.best_params_


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    X = preprocessor.transform(train[FEATURES])
    z = to_target(train["Yield"], train["Yield_lag1"])
    print("Training data shape:", X.shape)

    print("Tuning Random Forest...")
    rf_params = tune(RandomForestRegressor(random_state=42, n_jobs=N_JOBS), {
        "n_estimators": [200, 300],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": [0.3, 0.5, 0.8],
        "max_depth": [10, 20, None],
    }, X, z)
    print("Best:", rf_params)

    print("Tuning XGBoost...")
    xgb_params = tune(XGBRegressor(random_state=42, n_jobs=N_JOBS), {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }, X, z)
    print("Best:", xgb_params)

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
    for file_name, model in models.items():
        model.fit(X, z)
        joblib.dump(model, os.path.join(MODELS_DIR, file_name + ".joblib"))
        print("Saved", MODELS[file_name])
