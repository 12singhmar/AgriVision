# first look at the raw data

import os
import pandas as pd

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "crop_yield.csv")
df = pd.read_csv(path)

print(df.shape)
print(df.head())
print(df.dtypes)

print("\nmissing values:")
print(df.isnull().sum())
print("duplicates:", df.duplicated().sum())

# some text values have extra spaces at the end
print("\nCrop values with extra spaces:", (df["Crop"] != df["Crop"].str.strip()).sum())
print("Season values with extra spaces:", (df["Season"] != df["Season"].str.strip()).sum())

print("\nunique crops:", df["Crop"].str.strip().nunique())
print("unique states:", df["State"].nunique())
print("unique seasons:", df["Season"].str.strip().nunique())
print("years:", df["Crop_Year"].min(), "to", df["Crop_Year"].max())

print(df.describe())

# is yield just production / area?
ratio = df["Yield"] / (df["Production"] / df["Area"])
print("\nyield / (production/area) median:", ratio.median())

# fertilizer and pesticide vs area
print("\ncorrelation with area:")
print(df[["Area", "Fertilizer", "Pesticide"]].corr())

# coconut has really big yield numbers
print("\nmean yield by crop (top 5):")
print(df.groupby("Crop")["Yield"].mean().sort_values(ascending=False).head())

print("\nrows per year:")
print(df["Crop_Year"].value_counts().sort_index())
