# AgriVision streamlit app
# run: streamlit run app/app.py

import os
import sys
import pandas as pd
import streamlit as st

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(os.path.join(BASE_DIR, "src"))
from prediction import predict_yield

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

st.set_page_config(page_title="AgriVision", layout="wide")


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


df = load_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"))

st.sidebar.title("AgriVision")
page = st.sidebar.radio("Page", ["Data", "Predict", "Results"])
st.sidebar.write("Crop yield forecasting for Indian states (1997-2020)")


if page == "Data":
    st.title("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows used", len(df))
    c2.metric("Crops", df["Crop"].nunique())
    c3.metric("States", df["State"].nunique())
    c4.metric("Years", str(df["Crop_Year"].min()) + " - " + str(df["Crop_Year"].max()))

    st.write("Source: Kaggle - Crop Yield in Indian States. One row = one crop, state, season and year.")
    st.subheader("What I cleaned")
    st.write("- Removed extra spaces in Season and Crop names")
    st.write("- Removed Coconut (its yield is in nuts/ha, others are in tonnes/ha)")
    st.write("- Did not use Production and Area (Yield = Production / Area)")
    st.write("- Did not use Fertilizer and Pesticide (they are just Area x a fixed number per year)")
    st.write("- Added last year's yield, 3 year average and years since last record")
    st.write("- Train = before 2017, Test = 2017 to 2020")

    st.subheader("Sample rows")
    st.dataframe(df.head(50))
    st.subheader("Most common crops")
    st.bar_chart(df["Crop"].value_counts().head(15))


elif page == "Predict":
    st.title("Predict next year's yield")
    crops = sorted(df["Crop"].unique())
    states = sorted(df["State"].unique())
    seasons = sorted(df["Season"].unique())

    c1, c2, c3 = st.columns(3)
    crop = c1.selectbox("Crop", crops, index=crops.index("Wheat"))
    state = c2.selectbox("State", states, index=states.index("Punjab"))
    season = c3.selectbox("Season", seasons, index=seasons.index("Rabi"))

    history = df[(df["Crop"] == crop) & (df["State"] == state) & (df["Season"] == season)]
    history = history.sort_values("Crop_Year")
    if len(history) == 0:
        st.warning("No past data for this combination, so it can't be predicted. Try another one.")
        st.stop()

    last = history.iloc[-1]
    st.write("Years of data found:", len(history), "| latest year:", int(last["Crop_Year"]),
             "| yield:", round(last["Yield"], 2), "t/ha")
    st.line_chart(history.set_index("Crop_Year")["Yield"])

    c1, c2, c3 = st.columns(3)
    year = c1.number_input("Year to predict", 1998, 2030, int(last["Crop_Year"]) + 1)
    lag1 = c2.number_input("Last year's yield (t/ha)", 0.0, value=float(last["Yield"]))
    roll3 = c3.number_input("Average of last 3 years (t/ha)", 0.0, value=float(history["Yield"].tail(3).mean()))
    rain = st.slider("Annual rainfall (mm)", 300.0, 6600.0, float(last["Annual_Rainfall"]))

    model_files = {
        "XGBoost": "xgboost",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "Ridge Regression": "ridge_regression",
        "Linear Regression": "linear_regression",
    }
    model_name = st.selectbox("Model", list(model_files.keys()))

    if st.button("Predict"):
        gap = int(year) - int(last["Crop_Year"])
        if gap < 1:
            gap = 1
        row = {"Crop": crop, "State": state, "Season": season, "Crop_Year": int(year),
               "Annual_Rainfall": rain, "Yield_lag1": lag1, "Yield_roll3": roll3,
               "Years_since_last": gap}
        pred = predict_yield(row, model_files[model_name])

        c1, c2 = st.columns(2)
        c1.metric("Predicted yield (t/ha)", round(pred, 3))
        c2.metric("Last year's yield (t/ha)", round(lag1, 3))
        if lag1 > 0:
            change = (pred / lag1 - 1) * 100
            st.write("Change from last year:", round(change, 1), "%")


else:
    st.title("Results on 2017-2020 (not used in training)")
    results = load_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))
    st.dataframe(results)

    st.write("XGBoost is the only model that beats both baselines on MAE, MSE, RMSE and R2.")
    st.write("The last-year baseline still has a lower median % error, so for stable crops "
             "last year's value is already a good guess. The model helps most when yield changes a lot.")

    c1, c2 = st.columns(2)
    c1.image(os.path.join(FIGURES_DIR, "model_comparison_bars.png"))
    c2.image(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"))

    st.subheader("Error by crop (10 most common crops)")
    st.dataframe(load_csv(os.path.join(REPORTS_DIR, "crop_errors.csv")))

    st.subheader("Feature importance (SHAP)")
    c1, c2 = st.columns(2)
    c1.image(os.path.join(FIGURES_DIR, "shap_importance_bar.png"))
    c2.dataframe(load_csv(os.path.join(REPORTS_DIR, "shap_feature_importance.csv")))

    st.subheader("Punjab (only 56 test rows)")
    st.dataframe(load_csv(os.path.join(REPORTS_DIR, "punjab_metrics.csv")))
    st.image(os.path.join(FIGURES_DIR, "punjab_historical_trends.png"))
