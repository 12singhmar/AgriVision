# predicts yield for one input (used in the streamlit app)

import os
import joblib
import pandas as pd

from preprocessing import MODELS_DIR, FEATURES, from_target


def predict_yield(row, model_file="xgboost"):
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, model_file + ".joblib"))

    X = preprocessor.transform(pd.DataFrame([row])[FEATURES])
    z = model.predict(X)[0]
    return float(from_target(z, row["Yield_lag1"]))


if __name__ == "__main__":
    # quick test
    row = {"Crop": "Wheat", "State": "Punjab", "Season": "Rabi", "Crop_Year": 2019,
           "Annual_Rainfall": 650.0, "Yield_lag1": 5.07, "Yield_roll3": 4.99, "Years_since_last": 1}
    print(predict_yield(row))
