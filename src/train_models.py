"""
AgriVision - Model Training & Hyperparameter Tuning
====================================================
Trains and compares 7 regression models:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 Regularized Linear Model)
3. Random Forest Regressor (Bagging Tree Ensemble)
4. Extra Trees Regressor (Extremely Randomized Trees)
5. Gradient Boosting Regressor (Sequential Boosting)
6. XGBoost Regressor (Extreme Gradient Boosting)
7. CatBoost Regressor (Categorical Boosting)

Performs TimeSeriesSplit-aware RandomizedSearchCV on selected strong models
(Random Forest, XGBoost, CatBoost).
Saves all trained models into the models/ directory.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_processed_data():
    """Loads preprocessed numpy arrays from data/processed."""
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train_proc.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_proc.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))
    logging.info(f"Loaded processed data: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def tune_hyperparameters(X_train, y_train):
    """
    Performs focused RandomizedSearchCV using TimeSeriesSplit cross-validation
    on selected high-performing models: Random Forest, XGBoost, and CatBoost.
    """
    logging.info("Starting hyperparameter tuning using TimeSeriesSplit (3 splits)...")
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = {}

    # 1. Random Forest Tuning
    rf_param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [15, 25, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": [0.6, 0.8, 1.0],
    }
    logging.info("Tuning Random Forest Regressor...")
    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=6,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    best_params["Random Forest"] = rf_search.best_params_
    logging.info(f"Best Random Forest Params: {rf_search.best_params_}")

    # 2. XGBoost Tuning
    xgb_param_dist = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.08, 0.15],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    }
    logging.info("Tuning XGBoost Regressor...")
    xgb_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions=xgb_param_dist,
        n_iter=6,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    best_params["XGBoost"] = xgb_search.best_params_
    logging.info(f"Best XGBoost Params: {xgb_search.best_params_}")

    # 3. CatBoost Tuning
    cb_param_dist = {
        "iterations": [100, 150, 200],
        "learning_rate": [0.03, 0.08, 0.15],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
    }
    logging.info("Tuning CatBoost Regressor...")
    cb_search = RandomizedSearchCV(
        estimator=CatBoostRegressor(random_state=42, verbose=0),
        param_distributions=cb_param_dist,
        n_iter=6,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    cb_search.fit(X_train, y_train)
    best_params["CatBoost"] = cb_search.best_params_
    logging.info(f"Best CatBoost Params: {cb_search.best_params_}")

    # Save tuning summary
    tuning_path = os.path.join(REPORTS_DIR, "hyperparameter_tuning.json")
    with open(tuning_path, "w") as f:
        json.dump(best_params, f, indent=4)
    logging.info(f"Saved best hyperparameters to: {tuning_path}")

    return best_params


def train_and_save_all_models(X_train, y_train, best_params=None):
    """
    Instantiates and trains all 7 regression models with their optimized configurations.
    Saves each trained model artifact.
    """
    if best_params is None:
        best_params = {}

    rf_params = best_params.get(
        "Random Forest",
        {"n_estimators": 150, "max_depth": 25, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 0.8},
    )
    xgb_params = best_params.get(
        "XGBoost",
        {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.8},
    )
    cb_params = best_params.get(
        "CatBoost",
        {"iterations": 150, "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3},
    )

    models = {
        "linear_regression": (
            "Linear Regression",
            LinearRegression(),
        ),
        "ridge_regression": (
            "Ridge Regression",
            Ridge(alpha=10.0),
        ),
        "random_forest": (
            "Random Forest",
            RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1),
        ),
        "extra_trees": (
            "Extra Trees",
            ExtraTreesRegressor(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1),
        ),
        "gradient_boosting": (
            "Gradient Boosting",
            GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        ),
        "xgboost": (
            "XGBoost",
            XGBRegressor(**xgb_params, random_state=42, n_jobs=-1),
        ),
        "catboost": (
            "CatBoost",
            CatBoostRegressor(**cb_params, random_state=42, verbose=0),
        ),
    }

    trained_models = {}
    logging.info("=" * 60)
    logging.info("TRAINING ALL 7 REGRESSION MODELS")
    logging.info("=" * 60)

    for slug, (display_name, model) in models.items():
        logging.info(f"Training {display_name}...")
        model.fit(X_train, y_train)
        model_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(model, model_path)
        logging.info(f"Successfully trained and saved {display_name} -> {model_path}")
        trained_models[slug] = model

    return trained_models


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_processed_data()
    best_params = tune_hyperparameters(X_train, y_train)
    trained_models = train_and_save_all_models(X_train, y_train, best_params)
    print("\n[SUCCESS] All 7 models trained and saved successfully.")
