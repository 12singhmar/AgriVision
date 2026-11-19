import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AgriVision: Machine Learning-Based Crop Yield Prediction and Agricultural Performance Analysis\n",
                "**Exploratory Data Analysis, Leakage Elimination, Model Comparison & Explainable AI**\n",
                "\n",
                "### Objectives:\n",
                "1. Inspect and clean historical Indian crop yield records (1997–2020).\n",
                "2. Prevent Target Data Leakage (demonstrate why Production and Area must be excluded from predictive features).\n",
                "3. Train and compare 7 regression algorithms using a chronological train/test split.\n",
                "4. Hyperparameter tuning on selected tree ensembles.\n",
                "5. Explainable AI with SHAP.\n",
                "6. Focused agricultural analysis of Punjab."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sns.set_theme(style='whitegrid')\n",
                "print('Libraries successfully imported!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Loading & Hygiene Inspection\n",
                "Load raw Kaggle dataset and inspect dimensions, column types, null counts, and string whitespace."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_path = '../data/raw/crop_yield.csv'\n",
                "df = pd.read_csv(data_path)\n",
                "print('Dataset dimensions:', df.shape)\n",
                "display(df.head())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Strip whitespace in categoricals\n",
                "for col in ['Crop', 'Season', 'State']:\n",
                "    df[col] = df[col].astype(str).str.strip()\n",
                "\n",
                "print('Missing Values:\\n', df.isnull().sum())\n",
                "print('Duplicate Rows:', df.duplicated().sum())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Target Data Leakage Analysis\n",
                "Yield is defined mathematically as Production / Area. Inspecting their correlation reveals why they must be excluded."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "calc_ratio = df['Production'] / df['Area'].replace(0, np.nan)\n",
                "diff = (df['Yield'] - calc_ratio).abs()\n",
                "print('Median absolute difference between Yield and Production/Area:', diff.median())\n",
                "print('Exact match percentage:', (np.isclose(df['Yield'], calc_ratio, rtol=1e-3, atol=1e-3).sum() / len(df)) * 100)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Chronological Train/Test Split\n",
                "Train on historical records before 2017 (< 2017) and evaluate on subsequent years (>= 2017)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_df = df[df['Crop_Year'] < 2017]\n",
                "test_df = df[df['Crop_Year'] >= 2017]\n",
                "print(f'Training Set: {len(train_df):,} records ({len(train_df)/len(df)*100:.1f}%)')\n",
                "print(f'Test Set: {len(test_df):,} records ({len(test_df)/len(df)*100:.1f}%)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Benchmarking Results\n",
                "Review the held-out test evaluation results for all 7 regression models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "comp_df = pd.read_csv('../reports/model_comparison.csv')\n",
                "display(comp_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Punjab Regional Agricultural Analysis\n",
                "Evaluate model performance specifically on the Punjab subset."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "punjab_df = pd.read_csv('../reports/punjab_metrics.csv')\n",
                "display(punjab_df)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(r"C:\Users\12sin\.gemini\antigravity\scratch\AgriVision\notebooks\exploratory_analysis.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Notebook generated successfully!")
