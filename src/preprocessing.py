"""
AgriVision - Data Preprocessing
===============================
1. Clean the raw Kaggle CSV (strip whitespace, drop nulls/duplicates).
2. Remove Coconut (yield measured in nuts/ha, not tonnes/ha like every other crop).
3. Build lag features from each Crop/State/Season's own history:
   - Yield_lag1       : yield in the previous recorded year
   - Yield_roll3      : mean yield of the previous 3 recorded years
   - Years_since_last : gap (in years) to that previous record
4. Chronological split: train < 2017, test >= 2017.
5. Fit OneHotEncoder + StandardScaler on train only and save everything.
6. Define the training target: log change in yield versus last year (see to_target).

Columns NOT used as features, and why:
- Production, Area : Yield = Production / Area, and Production is only known after harvest.
- Fertilizer, Pesticide : in this dataset they equal Area x a fixed yearly rate
  (Fertilizer / Area is identical for every row in a year), so they carry no real
  information about how much fertilizer or pesticide was used.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "crop_yield.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SPLIT_YEAR = 2017
GROUP_COLS = ["Crop", "State", "Season"]
CAT_COLS = ["Crop", "Season", "State"]
NUM_COLS = ["Crop_Year", "Annual_Rainfall", "Yield_lag1", "Yield_roll3", "Years_since_last"]
TARGET = "Yield"


# Models do not predict Yield directly. They predict the CHANGE from last year on a log
# scale: z = log(1 + Yield) - log(1 + Yield_lag1)  (roughly the % change).
# This puts all crops on the same scale (sugarcane ~70 t/ha vs pulses ~0.7 t/ha)
# and lets the model focus on "will this year be better or worse than last year?".
def to_target(yield_values, lag1_values):
    return np.log1p(yield_values) - np.log1p(lag1_values)


def from_target(z_values, lag1_values):
    return np.maximum(np.expm1(z_values + np.log1p(lag1_values)), 0.0)


def load_and_clean_data(csv_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df):,} rows from {csv_path}")

    for col in ["Crop", "Season", "State"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Check 1: Yield should equal Production / Area
    ratio = df["Yield"] / (df["Production"] / df["Area"].replace(0, np.nan))
    logging.info(f"Median of Yield / (Production / Area) = {ratio.median():.3f}")

    # Check 2: Fertilizer / Area is the same for every row in a year => synthetic column
    fert_rate_std = (df["Fertilizer"] / df["Area"]).groupby(df["Crop_Year"]).std().max()
    logging.info(f"Max within-year std of Fertilizer/Area = {fert_rate_std:.6f} (0 means synthetic)")

    n_coconut = (df["Crop"] == "Coconut").sum()
    df = df[df["Crop"] != "Coconut"].reset_index(drop=True)
    logging.info(f"Removed {n_coconut} Coconut rows (different yield unit). {len(df):,} rows remain.")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every Crop/State/Season series, use only EARLIER years.
    shift(1) makes sure a row never sees its own yield or any future yield.
    """
    df = df.sort_values(GROUP_COLS + ["Crop_Year"]).reset_index(drop=True)
    grouped = df.groupby(GROUP_COLS)

    df["Yield_lag1"] = grouped["Yield"].shift(1)
    df["Yield_roll3"] = grouped["Yield"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["Years_since_last"] = df["Crop_Year"] - grouped["Crop_Year"].shift(1)

    # The first year of every series has no history, so it cannot be used.
    before = len(df)
    df = df.dropna(subset=["Yield_lag1"]).reset_index(drop=True)
    logging.info(f"Dropped {before - len(df):,} rows with no previous year (first record of each series).")
    return df.sort_values("Crop_Year", kind="mergesort").reset_index(drop=True)


def prepare_datasets(df: pd.DataFrame, split_year: int = SPLIT_YEAR):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df.to_csv(os.path.join(PROCESSED_DATA_DIR, "cleaned_crop_yield.csv"), index=False)

    train_df = df[df["Crop_Year"] < split_year].reset_index(drop=True)
    test_df = df[df["Crop_Year"] >= split_year].reset_index(drop=True)
    logging.info(f"Train: {len(train_df):,} rows (<{split_year}) | Test: {len(test_df):,} rows (>={split_year})")

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ])
    X_train = preprocessor.fit_transform(train_df[NUM_COLS + CAT_COLS])
    X_test = preprocessor.transform(test_df[NUM_COLS + CAT_COLS])

    feature_names = NUM_COLS + preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_COLS).tolist()
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))

    metadata = {
        "split_year": split_year,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "all_feature_names": feature_names,
        "categories": {
            "crops": sorted(df["Crop"].unique().tolist()),
            "seasons": sorted(df["Season"].unique().tolist()),
            "states": sorted(df["State"].unique().tolist()),
        },
        # Ranges from the training split only: this is what the models have seen.
        "numerical_ranges": {
            col: {"min": float(train_df[col].min()), "max": float(train_df[col].max()),
                  "median": float(train_df[col].median())}
            for col in NUM_COLS + [TARGET]
        },
        "train_combos": sorted((train_df["Crop"] + "|" + train_df["State"] + "|" + train_df["Season"]).unique().tolist()),
    }
    with open(os.path.join(PROCESSED_DATA_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_data.csv"), index=False)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train_proc.npy"), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test_proc.npy"), X_test)
    # y = actual yield (for evaluation); z = log-change target (what models are trained on)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), train_df[TARGET].values)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), test_df[TARGET].values)
    np.save(os.path.join(PROCESSED_DATA_DIR, "z_train.npy"), to_target(train_df[TARGET].values, train_df["Yield_lag1"].values))
    logging.info(f"{X_train.shape[1]} features after encoding ({len(NUM_COLS)} numeric + one-hot).")


if __name__ == "__main__":
    data = add_lag_features(load_and_clean_data())
    prepare_datasets(data)
    print("\n[SUCCESS] Preprocessing complete.")
