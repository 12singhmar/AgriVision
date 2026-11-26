# training the 5 models

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
N_JOBS = 2  # -1 was crashing on my laptop

# file name : name shown in tables
MODELS = {
    "linear_regression": "Linear Regression",
    "ridge_regression": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}

if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    X = preprocessor.transform(train[FEATURES])
    z = to_target(train["Yield"], train["Yield_lag1"])
    print("X:", X.shape)

    # TimeSeriesSplit so validation is always on later years than training
    cv = TimeSeriesSplit(n_splits=3)

    # tuning random forest
    rf_params = {
        "n_estimators": [200, 300],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": [0.3, 0.5, 0.8],
        "max_depth": [10, 20, None],
    }
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=N_JOBS), rf_params,
                                   n_iter=8, cv=cv, scoring="neg_root_mean_squared_error",
                                   random_state=42, n_jobs=N_JOBS)
    rf_search.fit(X, z)
    print("best rf:", rf_search.best_params_)

    # tuning xgboost
    xgb_params = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42, n_jobs=N_JOBS), xgb_params,
                                    n_iter=8, cv=cv, scoring="neg_root_mean_squared_error",
                                    random_state=42, n_jobs=N_JOBS)
    xgb_search.fit(X, z)
    print("best xgb:", xgb_search.best_params_)

    with open(os.path.join(REPORTS_DIR, "hyperparameter_tuning.json"), "w") as f:
        json.dump({"Random Forest": rf_search.best_params_, "XGBoost": xgb_search.best_params_}, f, indent=4)

    lr = LinearRegression()
    lr.fit(X, z)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X, z)

    # best_estimator_ is already retrained on all the training data
    rf = rf_search.best_estimator_

    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                   subsample=0.8, loss="huber", random_state=42)
    gb.fit(X, z)

    xgb = xgb_search.best_estimator_

    joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.joblib"))
    joblib.dump(ridge, os.path.join(MODELS_DIR, "ridge_regression.joblib"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(gb, os.path.join(MODELS_DIR, "gradient_boosting.joblib"))
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.joblib"))
    print("models saved")
