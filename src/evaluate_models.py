"""
AgriVision - Model Evaluation & Reporting
==========================================
Evaluates all 7 regression models on the held-out test set (2017-2020) and Punjab subset.
Calculates:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)
Generates high-resolution visualization figures:
1. Model Comparison Bar Chart (MAE, RMSE, R²)
2. Actual vs Predicted Yield Scatter Plot (Best Model)
3. Residual Distribution & Q-Q Plot
4. Punjab Historical Trends & Crop Comparisons
5. Prediction Error Analysis by Crop Group
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_CONFIGS = [
    ("linear_regression", "Linear Regression"),
    ("ridge_regression", "Ridge Regression"),
    ("random_forest", "Random Forest"),
    ("extra_trees", "Extra Trees"),
    ("gradient_boosting", "Gradient Boosting"),
    ("xgboost", "XGBoost"),
    ("catboost", "CatBoost"),
]


def load_test_data():
    """Loads processed test set and raw test dataframe for metadata analysis."""
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_proc.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_data.csv"))
    cleaned_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "cleaned_crop_yield.csv"))
    return X_test, y_test, test_df, cleaned_df


def evaluate_all():
    """Evaluates all 7 saved models and outputs comparison reports and plots."""
    X_test, y_test, test_df, full_df = load_test_data()
    punjab_mask = test_df["State"] == "Punjab"
    logging.info(f"Total Test records: {len(test_df)}, Punjab Test records: {punjab_mask.sum()}")

    results = []
    punjab_results = []
    predictions_dict = {}

    for slug, display_name in MODEL_CONFIGS:
        model_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            continue

        model = joblib.load(model_path)
        preds = model.predict(X_test)
        predictions_dict[display_name] = preds

        # Overall Test Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": display_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4),
        })

        # Punjab Test Metrics
        if punjab_mask.sum() > 0:
            y_test_punjab = y_test[punjab_mask]
            preds_punjab = preds[punjab_mask]
            p_mae = mean_absolute_error(y_test_punjab, preds_punjab)
            p_rmse = root_mean_squared_error(y_test_punjab, preds_punjab)
            p_r2 = r2_score(y_test_punjab, preds_punjab)

            punjab_results.append({
                "Model": display_name,
                "Punjab_MAE": round(p_mae, 4),
                "Punjab_RMSE": round(p_rmse, 4),
                "Punjab_R2_Score": round(p_r2, 4),
            })

    # Save overall comparison
    results_df = pd.DataFrame(results).sort_values(by="R2_Score", ascending=False).reset_index(drop=True)
    comp_csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(comp_csv_path, index=False)
    logging.info(f"\n=== OVERALL TEST EVALUATION RESULTS ===\n{results_df.to_string(index=False)}")

    # Save Punjab comparison
    if punjab_results:
        punjab_df = pd.DataFrame(punjab_results).sort_values(by="Punjab_R2_Score", ascending=False).reset_index(drop=True)
        punjab_csv_path = os.path.join(REPORTS_DIR, "punjab_metrics.csv")
        punjab_df.to_csv(punjab_csv_path, index=False)
        logging.info(f"\n=== PUNJAB TEST EVALUATION RESULTS ===\n{punjab_df.to_string(index=False)}")

    best_model_name = results_df.iloc[0]["Model"]
    logging.info(f"Identified Best Performing Model: {best_model_name}")

    # Generate Figures
    generate_evaluation_plots(results_df, y_test, predictions_dict, best_model_name, test_df, full_df)

    return results_df, best_model_name


def generate_evaluation_plots(results_df, y_test, predictions_dict, best_model_name, test_df, full_df):
    """Creates publication-ready visualizations for reports and Streamlit."""
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = sns.color_palette("viridis", len(results_df))

    # R2 Score
    sns.barplot(data=results_df, x="Model", y="R2_Score", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Test R² Score (Higher is Better)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # MAE
    sns.barplot(data=results_df, x="Model", y="MAE", ax=axes[1], palette="Oranges_r")
    axes[1].set_title("Mean Absolute Error (Lower is Better)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MAE", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='black', fontweight='bold', fontsize=10)

    # RMSE
    sns.barplot(data=results_df, x="Model", y="RMSE", ax=axes[2], palette="Reds_r")
    axes[2].set_title("Root Mean Squared Error (Lower is Better)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("RMSE", fontsize=11)
    axes[2].tick_params(axis="x", rotation=45)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='black', fontweight='bold', fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, "model_comparison_bars.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logging.info(f"Saved: {chart_path}")

    # 2. Actual vs Predicted Yield Scatter Plot (Best Model)
    best_preds = predictions_dict[best_model_name]
    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, best_preds, alpha=0.45, color="#1f77b4", edgecolors="none", s=35)
    max_val = max(np.max(y_test), np.max(best_preds))
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect 1:1 Prediction Line")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual Yield (log scale)", fontsize=12, fontweight="bold")
    plt.ylabel(f"Predicted Yield ({best_model_name}, log scale)", fontsize=12, fontweight="bold")
    plt.title(f"Actual vs Predicted Crop Yield – {best_model_name}\n(Held-Out Test Set 2017–2020)", fontsize=14, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    act_pred_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
    plt.savefig(act_pred_path, dpi=300)
    plt.close()
    logging.info(f"Saved: {act_pred_path}")

    # 3. Residual Distribution
    residuals = y_test - best_preds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals, bins=50, kde=True, ax=ax1, color="#2ca02c")
    ax1.set_xlim(-100, 100)
    ax1.set_title(f"Residual Distribution (Zoomed ±100)\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Residual (Actual - Predicted)")

    ax2.scatter(best_preds, residuals, alpha=0.4, color="#9467bd", s=25)
    ax2.axhline(0, color="red", linestyle="--", lw=1.5)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-100, 100)
    ax2.set_title("Residuals vs Predicted Yield (Zoomed)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Yield")
    ax2.set_ylabel("Residual")

    plt.tight_layout()
    resid_path = os.path.join(FIGURES_DIR, "residuals_distribution.png")
    plt.savefig(resid_path, dpi=300)
    plt.close()
    logging.info(f"Saved: {resid_path}")

    # 4. Punjab Analysis Plots
    punjab_df = full_df[full_df["State"] == "Punjab"].copy()
    if len(punjab_df) > 0:
        # Top Punjab Crops by record count
        top_punjab_crops = punjab_df["Crop"].value_counts().head(6).index.tolist()
        punjab_top = punjab_df[punjab_df["Crop"].isin(top_punjab_crops)]

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=punjab_top,
            x="Crop_Year",
            y="Yield",
            hue="Crop",
            marker="o",
            palette="tab10",
            errorbar=None
        )
        plt.title("Punjab Crop Yield Trends Over Time (1997–2019)", fontsize=14, fontweight="bold")
        plt.xlabel("Year", fontsize=12, fontweight="bold")
        plt.ylabel("Yield (Production per Hectare)", fontsize=12, fontweight="bold")
        plt.legend(title="Crop", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        punjab_trend_path = os.path.join(FIGURES_DIR, "punjab_historical_trends.png")
        plt.savefig(punjab_trend_path, dpi=300)
        plt.close()
        logging.info(f"Saved: {punjab_trend_path}")

        # Punjab Boxplot comparison
        plt.figure(figsize=(12, 6))
        order = punjab_top.groupby("Crop")["Yield"].median().sort_values(ascending=False).index
        sns.boxplot(data=punjab_top, x="Crop", y="Yield", order=order, palette="Set2")
        plt.title("Punjab Crop Yield Variability Across Top Crops", fontsize=14, fontweight="bold")
        plt.xlabel("Crop", fontsize=12, fontweight="bold")
        plt.ylabel("Yield", fontsize=12, fontweight="bold")
        plt.xticks(rotation=20)
        plt.tight_layout()
        punjab_box_path = os.path.join(FIGURES_DIR, "punjab_crop_yield_comparison.png")
        plt.savefig(punjab_box_path, dpi=300)
        plt.close()
        logging.info(f"Saved: {punjab_box_path}")


if __name__ == "__main__":
    evaluate_all()
    print("\n[SUCCESS] Model evaluation and reporting completed.")
