"""
AgriVision - Prediction
=======================
Predicts yield for one Crop/State/Season/Year using a trained model.
The user (or the app) supplies last year's yield and the 3-year average; the model
predicts the change from last year, which is converted back to t/ha.
"""

import os
import json
import joblib
import pandas as pd

from preprocessing import NUM_COLS, CAT_COLS, from_target

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


class AgriVisionPredictor:
    def __init__(self):
        with open(os.path.join(PROCESSED_DATA_DIR, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        self.models = {}

    def get_model(self, slug):
        if slug not in self.models:
            self.models[slug] = joblib.load(os.path.join(MODELS_DIR, f"{slug}.joblib"))
        return self.models[slug]

    def check_inputs(self, row: dict) -> list:
        """Raise on unknown categories; return warnings for inputs outside the training data."""
        cats = self.metadata["categories"]
        for key, allowed in [("Crop", cats["crops"]), ("Season", cats["seasons"]), ("State", cats["states"])]:
            if row[key] not in allowed:
                raise ValueError(f"Unknown {key}: {row[key]}")

        warnings = []
        if f"{row['Crop']}|{row['State']}|{row['Season']}" not in set(self.metadata["train_combos"]):
            warnings.append("This Crop/State/Season combination never appears in the training data.")
        for col in ["Annual_Rainfall", "Yield_lag1", "Yield_roll3"]:
            r = self.metadata["numerical_ranges"][col]
            if not r["min"] <= row[col] <= r["max"]:
                warnings.append(f"{col} = {row[col]:,.2f} is outside the training range [{r['min']:,.2f}, {r['max']:,.2f}].")
        return warnings

    def predict(self, row: dict, model_slug: str = "xgboost") -> dict:
        warnings = self.check_inputs(row)
        X = self.preprocessor.transform(pd.DataFrame([row])[NUM_COLS + CAT_COLS])
        z = self.get_model(model_slug).predict(X)[0]
        pred = float(from_target([z], [row["Yield_lag1"]])[0])
        return {"predicted_yield": round(pred, 3), "change_pct": round(100 * (pred / row["Yield_lag1"] - 1), 1)
                if row["Yield_lag1"] > 0 else None, "warnings": warnings}


if __name__ == "__main__":
    p = AgriVisionPredictor()
    print(p.predict({"Crop": "Wheat", "State": "Punjab", "Season": "Rabi", "Crop_Year": 2019,
                     "Annual_Rainfall": 650.0, "Yield_lag1": 5.07, "Yield_roll3": 4.99, "Years_since_last": 1}))
