# AgriVision - predict one row with a saved model (used by the Streamlit app)

import os
import joblib
import pandas as pd

from preprocessing import MODELS_DIR, FEATURES, from_target


def predict_yield(row, model_file="xgboost"):
    """row = dict with Crop, State, Season, Crop_Year, Annual_Rainfall,
    Yield_lag1, Yield_roll3, Years_since_last. Returns predicted yield in t/ha."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, model_file + ".joblib"))

    X = preprocessor.transform(pd.DataFrame([row])[FEATURES])
    z = model.predict(X)[0]                       # predicted log change
    return float(from_target(z, row["Yield_lag1"]))  # back to t/ha


if __name__ == "__main__":
    example = {"Crop": "Wheat", "State": "Punjab", "Season": "Rabi", "Crop_Year": 2019,
               "Annual_Rainfall": 650.0, "Yield_lag1": 5.07, "Yield_roll3": 4.99, "Years_since_last": 1}
    print("Predicted yield:", round(predict_yield(example), 3), "t/ha")
