"""
AgriVision - Data Preprocessing Pipeline
=========================================
Handles data loading, hygiene (whitespace trimming), data leakage prevention,
chronological train-test splitting, feature scaling, and categorical encoding.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "crop_yield.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SPLIT_YEAR = 2017  # Chronological split: < 2017 for training, >= 2017 for testing


def load_and_clean_data(csv_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw CSV and performs data hygiene:
    1. Trims extraneous whitespace from string/categorical columns.
    2. Validates data types and nulls.
    3. Analyzes and documents data leakage between (Production, Area) and Yield.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw data file not found at: {csv_path}")

    logging.info(f"Loading raw dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    logging.info(f"Dataset loaded successfully: {df.shape[0]:,} rows, {df.shape[1]} columns.")

    # 1. Clean categorical whitespace
    string_cols = ["Crop", "Season", "State"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 2. Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logging.warning(f"Found missing values:\n{null_counts[null_counts > 0]}")
        df = df.dropna().reset_index(drop=True)
    else:
        logging.info("Zero missing values found in raw data.")

    # 3. Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logging.info(f"Dropping {dup_count} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logging.info("Zero duplicate records found.")

    # 4. Verify Data Leakage Relationship
    # Yield is mathematically derived from (Production / Area)
    calc_ratio = df["Production"] / df["Area"].replace(0, np.nan)
    ratio_diff = (df["Yield"] - calc_ratio).abs()
    median_diff = ratio_diff.median()
    logging.info(
        f"Data Leakage Check: Median absolute difference between Yield and (Production / Area) is {median_diff:.4f}. "
        "Production and Area MUST be excluded from predictive features to prevent trivial leakage."
    )

    return df


def prepare_datasets(df: pd.DataFrame, split_year: int = SPLIT_YEAR):
    """
    Prepares train and test splits chronologically:
    - Train: Crop_Year < split_year (earlier historical records)
    - Test: Crop_Year >= split_year (future held-out years)

    Features selected for modeling (leakage-free):
    - Categorical: Crop, Season, State
    - Numerical: Crop_Year, Annual_Rainfall, Fertilizer, Pesticide
    Target:
    - Yield
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save full cleaned dataset
    cleaned_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_crop_yield.csv")
    df.to_csv(cleaned_path, index=False)
    logging.info(f"Cleaned dataset saved to: {cleaned_path}")

    # Chronological Split
    train_df = df[df["Crop_Year"] < split_year].copy().reset_index(drop=True)
    test_df = df[df["Crop_Year"] >= split_year].copy().reset_index(drop=True)

    logging.info(
        f"Chronological split at year {split_year}: "
        f"Train={len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%), "
        f"Test={len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)"
    )

    feature_cols = ["Crop", "Season", "State", "Crop_Year", "Annual_Rainfall", "Fertilizer", "Pesticide"]
    target_col = "Yield"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Preprocessing Pipeline: OneHotEncoder for categoricals + StandardScaler for numericals
    cat_cols = ["Crop", "Season", "State"]
    num_cols = ["Crop_Year", "Annual_Rainfall", "Fertilizer", "Pesticide"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    logging.info("Fitting preprocessor on training data only...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    # Save preprocessor artifact
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logging.info(f"Saved fitted preprocessor to: {preprocessor_path}")

    # Save metadata (feature names, supported ranges, category lists)
    metadata = {
        "split_year": split_year,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "all_feature_names": all_feature_names,
        "categories": {
            "crops": sorted(df["Crop"].unique().tolist()),
            "seasons": sorted(df["Season"].unique().tolist()),
            "states": sorted(df["State"].unique().tolist()),
        },
        "numerical_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
            }
            for col in num_cols + ["Yield"]
        },
    }

    metadata_path = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved dataset metadata to: {metadata_path}")

    # Save train and test sets to CSV
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_data.csv"), index=False)

    # Save preprocessed numpy arrays for fast model training
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train_proc.npy"), X_train_proc)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test_proc.npy"), X_test_proc)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train.values)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test.values)

    logging.info(
        f"Transformed features dimension: {X_train_proc.shape[1]} features "
        f"({len(num_cols)} numerical + {len(cat_feature_names)} one-hot encoded categorical)."
    )

    return {
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "feature_names": all_feature_names,
        "metadata": metadata,
    }


if __name__ == "__main__":
    df = load_and_clean_data()
    artifacts = prepare_datasets(df)
    print("\n[SUCCESS] Preprocessing pipeline executed successfully.")
