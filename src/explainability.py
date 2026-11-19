"""
AgriVision - Explainable AI (XAI) using SHAP
=============================================
Computes Shapley Additive exPlanations (SHAP) for the best tree-based model.
Generates:
1. SHAP Summary (Beeswarm) Plot
2. SHAP Global Feature Importance Bar Plot
3. Individual prediction waterfall / force explanation
4. Plain-language interpretation distinguishing association from causation.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def run_shap_analysis(sample_size: int = 500):
    """
    Computes SHAP values on a representative test sample using TreeExplainer.
    Saves summary plots and top feature importance rankings.
    """
    # Load metadata to get feature names
    meta_path = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_names = meta["all_feature_names"]

    # Load test data
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test_proc.npy"))
    logging.info(f"Loaded X_test with shape {X_test.shape}")

    # Select best model: prefer extra_trees or random_forest or xgboost
    preferred_models = ["extra_trees", "random_forest", "xgboost"]
    model_name = None
    model = None

    for m in preferred_models:
        path = os.path.join(MODELS_DIR, f"{m}.joblib")
        if os.path.exists(path):
            model = joblib.load(path)
            model_name = m
            break

    if model is None:
        raise FileNotFoundError("No trained tree model found in models/ directory.")

    logging.info(f"Using {model_name} for SHAP explainability analysis...")

    # Sample test set for efficient, accurate SHAP computation
    np.random.seed(42)
    sample_indices = np.random.choice(X_test.shape[0], size=min(sample_size, X_test.shape[0]), replace=False)
    X_sample = X_test[sample_indices]

    # Initialize TreeExplainer
    logging.info("Calculating TreeExplainer SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # 1. SHAP Global Feature Importance Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title(f"SHAP Global Feature Importance (Top 15)\nModel: {model_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "shap_importance_bar.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    logging.info(f"Saved: {bar_path}")

    # 2. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title(f"SHAP Feature Impact on Yield Prediction\nModel: {model_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    summary_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    logging.info(f"Saved: {summary_path}")

    # 3. Calculate mean absolute SHAP values for ranking table
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:15]
    ranking = [
        {
            "Rank": int(i + 1),
            "Feature": feature_names[idx],
            "Mean_Abs_SHAP": float(round(mean_abs_shap[idx], 4)),
            "Agronomic_Role": get_agronomic_role(feature_names[idx]),
        }
        for i, idx in enumerate(top_indices)
    ]

    ranking_path = os.path.join(REPORTS_DIR, "shap_feature_importance.json")
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=4)
    logging.info(f"Saved feature importance ranking to: {ranking_path}")

    # Save SHAP explainer sample for fast interactive Streamlit waterfall plots
    shap_artifact = {
        "expected_value": float(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]),
        "feature_names": feature_names,
        "sample_X": X_sample[:50].tolist(),
        "sample_shap": shap_values[:50].tolist(),
    }
    with open(os.path.join(REPORTS_DIR, "shap_sample_data.json"), "w") as f:
        json.dump(shap_artifact, f)

    return ranking


def get_agronomic_role(feature_name: str) -> str:
    """Provides plain-language agronomic context for feature influence."""
    if "Crop_Coconut" in feature_name:
        return "Biological crop type baseline: Coconuts yield thousands of nuts per hectare (vastly higher raw unit count than grains)."
    elif "Crop_Sugarcane" in feature_name:
        return "High fresh biomass crop: Sugarcane yields 40–100 tonnes of stalks per hectare."
    elif "Fertilizer" in feature_name:
        return "Macronutrient availability: Higher input corresponds with intensive cultivation zones, though association is not direct causation."
    elif "Pesticide" in feature_name:
        return "Crop protection intensity: Correlated with commercial cash crops requiring disease/pest suppression."
    elif "Annual_Rainfall" in feature_name:
        return "Moisture & hydrologic regime: Dictates water availability for rainfed and semi-irrigated agro-climatic zones."
    elif "Crop_Year" in feature_name:
        return "Temporal technology proxy: Captures multi-decade agronomic progress, mechanization, and hybrid seed adoption."
    elif "State" in feature_name:
        return "Regional agro-climatic & soil suitability: Reflects local irrigation infrastructure, soil quality, and farming policies."
    elif "Season" in feature_name:
        return "Photoperiod & temperature cycle: Kharif (monsoon) vs Rabi (winter dry season) crop phenology."
    else:
        return "Specific crop genotype or regional indicator feature."


if __name__ == "__main__":
    run_shap_analysis()
    print("\n[SUCCESS] SHAP explainability analysis completed.")
