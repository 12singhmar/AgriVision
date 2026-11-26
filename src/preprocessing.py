# cleaning + lag features + train/test split

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

SPLIT_YEAR = 2017
CAT_COLS = ["Crop", "Season", "State"]
NUM_COLS = ["Crop_Year", "Annual_Rainfall", "Yield_lag1", "Yield_roll3", "Years_since_last"]
FEATURES = NUM_COLS + CAT_COLS


# model predicts change from last year (log scale) instead of the yield itself
# because sugarcane is ~70 t/ha and pulses are <1 t/ha
def to_target(yield_now, yield_last):
    return np.log1p(yield_now) - np.log1p(yield_last)


# converts the prediction back to t/ha
def from_target(z, yield_last):
    pred = np.expm1(z + np.log1p(yield_last))
    return np.maximum(pred, 0)


def load_and_clean():
    df = pd.read_csv(RAW_FILE)
    print("raw data:", df.shape)

    # Season and some Crop names have spaces at the end
    df["Crop"] = df["Crop"].str.strip()
    df["Season"] = df["Season"].str.strip()
    df["State"] = df["State"].str.strip()

    df = df.dropna()
    df = df.drop_duplicates()

    # checking if fertilizer is real data
    # if fertilizer/area is the same for every row in a year then its just area * some number
    df["fert_per_area"] = df["Fertilizer"] / df["Area"]
    spread = df.groupby("Crop_Year")["fert_per_area"].std()
    print("max std of fertilizer/area within a year:", spread.max())
    df = df.drop(columns=["fert_per_area"])

    # coconut yield is in nuts/ha, rest are in tonnes/ha so removing it
    df = df[df["Crop"] != "Coconut"]
    df = df.reset_index(drop=True)
    print("after cleaning:", df.shape)
    return df


def add_lag_features(df):
    # one series = same crop, state and season over the years
    df = df.sort_values(["Crop", "State", "Season", "Crop_Year"]).reset_index(drop=True)
    groups = df.groupby(["Crop", "State", "Season"])

    # shift(1) gives the previous year's value in the same series
    df["Yield_lag1"] = groups["Yield"].shift(1)
    df["lag2"] = groups["Yield"].shift(2)
    df["lag3"] = groups["Yield"].shift(3)
    df["Yield_roll3"] = df[["Yield_lag1", "lag2", "lag3"]].mean(axis=1)  # mean skips NaN
    df["Years_since_last"] = df["Crop_Year"] - groups["Crop_Year"].shift(1)
    df = df.drop(columns=["lag2", "lag3"])

    # first year of every series has no previous year
    df = df.dropna(subset=["Yield_lag1"])
    print("after lag features:", df.shape)

    # sort by year for TimeSeriesSplit later (stable = same order every run)
    df = df.sort_values("Crop_Year", kind="stable").reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_and_clean()
    df = add_lag_features(df)

    train = df[df["Crop_Year"] < SPLIT_YEAR]
    test = df[df["Crop_Year"] >= SPLIT_YEAR]
    print("train:", train.shape, "test:", test.shape)

    # scaling numbers + one hot encoding text columns
    # fit only on train so test data doesnt leak in
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ], verbose_feature_names_out=False)
    preprocessor.fit(train[FEATURES])
    print("features after encoding:", len(preprocessor.get_feature_names_out()))

    df.to_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"), index=False)
    train.to_csv(os.path.join(DATA_DIR, "train_data.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test_data.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    print("saved")
