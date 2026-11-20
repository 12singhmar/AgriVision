# AgriVision - Step 1: cleaning, lag features, train/test split
#
# Columns I do NOT use as inputs:
#   Production, Area      -> Yield = Production / Area, so this would be cheating
#   Fertilizer, Pesticide -> they are just Area x a fixed number per year (not real data)
#   Coconut rows          -> yield is in nuts/ha, every other crop is in tonnes/ha

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "crop_yield.csv")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SPLIT_YEAR = 2017  # train on years before 2017, test on 2017-2020
CAT_COLS = ["Crop", "Season", "State"]
NUM_COLS = ["Crop_Year", "Annual_Rainfall", "Yield_lag1", "Yield_roll3", "Years_since_last"]
FEATURES = NUM_COLS + CAT_COLS


# The models don't predict Yield directly.
# They predict the change from last year on a log scale:
#     z = log(1 + Yield) - log(1 + last year's Yield)      (roughly the % change)
# so sugarcane (~70 t/ha) and pulses (~0.7 t/ha) are on the same scale.
def to_target(yield_now, yield_last):
    return np.log1p(yield_now) - np.log1p(yield_last)


def from_target(z, yield_last):
    pred = np.expm1(z + np.log1p(yield_last))
    return np.maximum(pred, 0)  # yield can't be negative


def load_and_clean():
    df = pd.read_csv(RAW_FILE)
    print("Raw rows:", len(df))

    # some text values have extra spaces at the end, e.g. "Kharif     "
    df["Crop"] = df["Crop"].str.strip()
    df["Season"] = df["Season"].str.strip()
    df["State"] = df["State"].str.strip()

    df = df.dropna()
    df = df.drop_duplicates()

    # check: is Fertilizer just Area x a yearly rate? (std of the rate within a year = 0)
    rate = df["Fertilizer"] / df["Area"]
    print("Max std of Fertilizer/Area inside one year:", round(rate.groupby(df["Crop_Year"]).std().max(), 6))

    df = df[df["Crop"] != "Coconut"]
    print("Rows after cleaning and removing Coconut:", len(df))
    return df.reset_index(drop=True)


def add_lag_features(df):
    # Each series = one Crop + State + Season, sorted by year.
    # shift(1) moves values down by one row, so each row only sees PAST years.
    df = df.sort_values(["Crop", "State", "Season", "Crop_Year"]).reset_index(drop=True)
    series = df.groupby(["Crop", "State", "Season"])

    df["Yield_lag1"] = series["Yield"].shift(1)
    df["Yield_roll3"] = series["Yield"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["Years_since_last"] = df["Crop_Year"] - series["Crop_Year"].shift(1)

    # the first year of every series has no "last year", so drop it
    df = df.dropna(subset=["Yield_lag1"])
    print("Rows with history:", len(df))

    # sort by year (needed later for TimeSeriesSplit)
    return df.sort_values("Crop_Year", kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = add_lag_features(load_and_clean())

    train = df[df["Crop_Year"] < SPLIT_YEAR]
    test = df[df["Crop_Year"] >= SPLIT_YEAR]
    print("Train rows:", len(train), "| Test rows:", len(test))

    # scale numbers, one-hot encode text columns. Fit on TRAIN only.
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ], verbose_feature_names_out=False)
    preprocessor.fit(train[FEATURES])
    print("Number of features after encoding:", len(preprocessor.get_feature_names_out()))

    df.to_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"), index=False)
    train.to_csv(os.path.join(DATA_DIR, "train_data.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test_data.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    print("Done.")
