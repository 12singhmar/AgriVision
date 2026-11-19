"""
AgriVision - Prediction Service
================================
Provides single-point and batch prediction interfaces with:
1. Strict input validation against training categories and numerical ranges.
2. Extrapolation detection and warning generation.
3. Preprocessing consistency using the fitted ColumnTransformer artifact.
4. Model selection across all 7 trained algorithms.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


class AgriVisionPredictor:
    """Predictor class encapsulating preprocessor, models, and range checks."""

    def __init__(self, default_model_slug: str = "extra_trees"):
        self.meta_path = os.path.join(PROCESSED_DATA_DIR, "metadata.json")
        self.preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata not found at: {self.meta_path}. Run preprocessing first.")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at: {self.preprocessor_path}.")

        with open(self.meta_path, "r") as f:
            self.metadata = json.load(f)

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.default_model_slug = default_model_slug
        self.loaded_models = {}

    def get_model(self, model_slug: str = None):
        """Loads and caches the requested model artifact."""
        slug = model_slug or self.default_model_slug
        if slug not in self.loaded_models:
            model_file = os.path.join(MODELS_DIR, f"{slug}.joblib")
            if not os.path.exists(model_file):
                # Fallback to random forest or extra trees if slug not found
                fallback = "random_forest.joblib" if os.path.exists(os.path.join(MODELS_DIR, "random_forest.joblib")) else "extra_trees.joblib"
                model_file = os.path.join(MODELS_DIR, fallback)
                slug = fallback.replace(".joblib", "")
            self.loaded_models[slug] = joblib.load(model_file)
        return self.loaded_models[slug], slug

    def validate_and_check_extrapolation(self, input_dict: dict):
        """
        Validates categories and checks whether numerical values lie outside
        the training distribution, generating actionable extrapolation warnings.
        """
        warnings = []
        categories = self.metadata["categories"]
        num_ranges = self.metadata["numerical_ranges"]

        # Category checks
        crop = str(input_dict.get("Crop", "")).strip()
        if crop not in categories["crops"]:
            raise ValueError(f"Unknown crop '{crop}'. Must be one of: {categories['crops']}")

        season = str(input_dict.get("Season", "")).strip()
        if season not in categories["seasons"]:
            raise ValueError(f"Unknown season '{season}'. Must be one of: {categories['seasons']}")

        state = str(input_dict.get("State", "")).strip()
        if state not in categories["states"]:
            raise ValueError(f"Unknown state '{state}'. Must be one of: {categories['states']}")

        # Numerical range and extrapolation checks
        for feature in ["Annual_Rainfall", "Fertilizer", "Pesticide", "Crop_Year"]:
            val = float(input_dict.get(feature, 0.0))
            f_min = num_ranges[feature]["min"]
            f_max = num_ranges[feature]["max"]

            if val < f_min or val > f_max:
                warnings.append(
                    f"Warning: {feature} value {val:,.2f} is outside the historical training range "
                    f"[{f_min:,.2f}, {f_max:,.2f}]. Model prediction may involve extrapolation uncertainty."
                )

        return warnings

    def predict_single(self, input_dict: dict, model_slug: str = None) -> dict:
        """
        Generates a yield prediction for a single user-specified agricultural scenario.
        Returns predicted yield, warning flags, and model metadata.
        """
        # Validate inputs and gather extrapolation warnings
        extrapolation_warnings = self.validate_and_check_extrapolation(input_dict)

        # Build single-row DataFrame
        df_row = pd.DataFrame([{
            "Crop": str(input_dict["Crop"]).strip(),
            "Season": str(input_dict["Season"]).strip(),
            "State": str(input_dict["State"]).strip(),
            "Crop_Year": int(input_dict["Crop_Year"]),
            "Annual_Rainfall": float(input_dict["Annual_Rainfall"]),
            "Fertilizer": float(input_dict["Fertilizer"]),
            "Pesticide": float(input_dict["Pesticide"]),
        }])

        # Transform features
        X_proc = self.preprocessor.transform(df_row)

        # Predict
        model, used_slug = self.get_model(model_slug)
        raw_pred = float(model.predict(X_proc)[0])
        pred_yield = max(0.0, raw_pred)  # Crop yield cannot be negative

        return {
            "predicted_yield": round(pred_yield, 4),
            "model_used": used_slug.replace("_", " ").title(),
            "model_slug": used_slug,
            "has_warnings": len(extrapolation_warnings) > 0,
            "warnings": extrapolation_warnings,
            "disclaimer": (
                "Notice: This prediction is a machine-learning statistical estimate based on historical Indian "
                "agricultural records (1997–2020). Actual real-world yield depends on local microclimates, seed varieties, "
                "soil micronutrients (NPK/pH), irrigation timing, and pest outbreaks."
            ),
        }


if __name__ == "__main__":
    # Test sample prediction
    predictor = AgriVisionPredictor()
    sample_input = {
        "Crop": "Wheat",
        "Season": "Rabi",
        "State": "Punjab",
        "Crop_Year": 2019,
        "Annual_Rainfall": 650.0,
        "Fertilizer": 4500000.0,
        "Pesticide": 9500.0,
    }
    result = predictor.predict_single(sample_input)
    print("\n[SUCCESS] Sample Prediction Output:")
    print(json.dumps(result, indent=2))
