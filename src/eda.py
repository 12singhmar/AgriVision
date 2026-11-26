# EDA graphs (run after preprocessing.py)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import RAW_FILE, DATA_DIR, BASE_DIR

FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

raw = pd.read_csv(RAW_FILE)
df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"))

# rows per year
counts = raw["Crop_Year"].value_counts().sort_index()
plt.figure(figsize=(9, 4))
plt.bar(counts.index, counts.values, color="green")
plt.title("Number of records per year")
plt.xlabel("Year")
plt.ylabel("Rows")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_rows_per_year.png"), dpi=150)
plt.close()

# most common crops
top = df["Crop"].value_counts().head(15)
plt.figure(figsize=(8, 5))
plt.barh(top.index[::-1], top.values[::-1], color="green")
plt.title("15 most common crops (after cleaning)")
plt.xlabel("Rows")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_top_crops.png"), dpi=150)
plt.close()

# yield distribution, log scale because values go from ~0 to 80+
y = df[df["Yield"] > 0]["Yield"]
plt.figure(figsize=(8, 4))
plt.hist(np.log10(y), bins=60, color="green")
plt.title("Distribution of yield")
plt.xlabel("log10(Yield in t/ha)   (0 = 1 t/ha, 1 = 10 t/ha, 2 = 100 t/ha)")
plt.ylabel("Rows")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_yield_distribution.png"), dpi=150)
plt.close()

# % change from last year
change = (df["Yield"] / df["Yield_lag1"] - 1) * 100
change = change.replace([np.inf, -np.inf], np.nan).dropna()
print("median change from last year (%):", change.abs().median())
plt.figure(figsize=(8, 4))
plt.hist(change.clip(-100, 100), bins=80, color="green")
plt.title("Change in yield compared with last year")
plt.xlabel("% change (cut off at -100 and +100)")
plt.ylabel("Rows")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_yearly_change.png"), dpi=150)
plt.close()

# correlation of the number columns
cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
corr = raw[cols].corr()
print(corr.round(2))
plt.figure(figsize=(7, 6))
plt.imshow(corr, cmap="Greens", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(6), cols, rotation=45)
plt.yticks(range(6), cols)
for i in range(6):
    for j in range(6):
        plt.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")
plt.title("Correlation (raw data)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_correlation.png"), dpi=150)
plt.close()
print("eda graphs saved")
