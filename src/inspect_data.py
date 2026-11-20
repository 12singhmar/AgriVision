import os
import pandas as pd
import numpy as np

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'crop_yield.csv')
df = pd.read_csv(csv_path)

print("=" * 60)
print("1. DATASET DIMENSIONS")
print(f"Total Rows: {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")
print("=" * 60)

print("\n2. COLUMNS & DATA TYPES")
for col, dtype in df.dtypes.items():
    print(f"  - {col}: {dtype}")

print("\n3. MISSING VALUES & DUPLICATES")
missing = df.isnull().sum()
for col, count in missing.items():
    pct = (count / len(df)) * 100
    print(f"  - {col}: {count} ({pct:.2f}%)")
print(f"Total Duplicated Rows: {df.duplicated().sum()}")

print("\n4. STRING WHITESPACE CHECK")
for col in ['Crop', 'Season', 'State']:
    untrimmed = df[col].dropna().apply(lambda x: len(x) != len(x.strip())).sum()
    print(f"  - {col}: {untrimmed} rows have leading/trailing whitespace")

print("\n5. UNIQUE COUNTS")
for col in ['Crop', 'Season', 'State', 'Crop_Year']:
    cleaned = df[col].astype(str).str.strip()
    print(f"  - Unique {col}: {cleaned.nunique()}")

print("\n6. SUMMARY STATISTICS OF NUMERICAL FEATURES")
num_cols = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
desc = df[num_cols].describe().T[['min', '25%', '50%', '75%', 'max', 'mean', 'std']]
print(desc.to_string())

print("\n7. INVESTIGATION: YIELD vs (PRODUCTION / AREA) DATA LEAKAGE")
calc_yield = df['Production'] / df['Area']
ratio = df['Yield'] / calc_yield.replace(0, np.nan)
print("Yield / (Production / Area) distribution:")
print(ratio.describe())
exact_match = np.isclose(df['Yield'], calc_yield, rtol=1e-3, atol=1e-3).sum()
print(f"Exact matches (within 0.1% tolerance): {exact_match} / {len(df)} ({exact_match/len(df)*100:.2f}%)")

print("\n8. CHRONOLOGICAL DISTRIBUTION (Crop_Year)")
year_counts = df['Crop_Year'].value_counts().sort_index()
print(year_counts.to_string())

print("\n9. PUNJAB FOCUS ANALYSIS")
df['State_Clean'] = df['State'].astype(str).str.strip()
df['Crop_Clean'] = df['Crop'].astype(str).str.strip()
punjab = df[df['State_Clean'] == 'Punjab']
print(f"Total Punjab Records: {len(punjab)}")
print(f"Punjab Crops ({punjab['Crop_Clean'].nunique()}): {sorted(punjab['Crop_Clean'].unique())}")
print(f"Punjab Crop Years ({punjab['Crop_Year'].nunique()}): {sorted(punjab['Crop_Year'].unique())}")
print(f"Punjab Seasons: {punjab['Season'].str.strip().value_counts().to_dict()}")

print("\n10. FERTILIZER & PESTICIDE RELATIONSHIP WITH AREA")
print("Correlation of Fertilizer with Area:", df['Fertilizer'].corr(df['Area']))
print("Correlation of Pesticide with Area:", df['Pesticide'].corr(df['Area']))
fert_per_area = df['Fertilizer'] / df['Area']
pest_per_area = df['Pesticide'] / df['Area']
print(f"Fertilizer / Area mean: {fert_per_area.mean():.4f}, std: {fert_per_area.std():.4f}, min: {fert_per_area.min():.4f}, max: {fert_per_area.max():.4f}")
print(f"Pesticide / Area mean: {pest_per_area.mean():.4f}, std: {pest_per_area.std():.4f}, min: {pest_per_area.min():.4f}, max: {pest_per_area.max():.4f}")
