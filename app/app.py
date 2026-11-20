# AgriVision - Streamlit app
# Run with:  streamlit run app/app.py
# Pages: Data / Predict / Results

import os
import sys
import pandas as pd
import streamlit as st

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(os.path.join(BASE_DIR, "src"))
from prediction import predict_yield
from train_models import MODELS

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

st.set_page_config(page_title="AgriVision", page_icon="🌾", layout="wide")


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


df = load_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"))

st.sidebar.title("🌾 AgriVision")
page = st.sidebar.radio("Page", ["Data", "Predict", "Results"])
st.sidebar.caption("Crop yield forecasting for Indian states, 1997–2020")

# ---------------- DATA ----------------
if page == "Data":
    st.title("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows used", len(df))
    c2.metric("Crops", df["Crop"].nunique())
    c3.metric("States", df["State"].nunique())
    c4.metric("Years", str(df["Crop_Year"].min()) + "–" + str(df["Crop_Year"].max()))

    st.markdown("""
**Source:** Kaggle "Crop Yield in Indian States" (one row per crop × season × state × year).

**Cleaning and decisions**
- Removed extra spaces from `Season` and `Crop`.
- Removed **Coconut**: its yield is in nuts/ha, every other crop is in tonnes/ha.
- **Production** and **Area** are not used: Yield = Production / Area.
- **Fertilizer** and **Pesticide** are not used: they are just Area × a fixed rate per year.
- Added **lag features**: last year's yield, 3-year average and years since the last record.
  The first year of each crop/state/season has no history and was dropped.
- Train on years before 2017, test on 2017–2020.
""")
    st.subheader("Sample rows")
    st.dataframe(df.head(50))
    st.subheader("Most common crops")
    st.bar_chart(df["Crop"].value_counts().head(15))

# ---------------- PREDICT ----------------
elif page == "Predict":
    st.title("Forecast next year's yield")
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
        st.warning("No history for this combination, so last year's yield is unknown. Pick another one.")
        st.stop()

    last = history.iloc[-1]
    st.caption(f"Found {len(history)} years of history. Latest: {int(last['Crop_Year'])}, "
               f"yield {last['Yield']:.2f} t/ha. The inputs below are filled from it.")
    st.line_chart(history.set_index("Crop_Year")["Yield"])

    c1, c2, c3 = st.columns(3)
    year = c1.number_input("Year to forecast", 1998, 2030, int(last["Crop_Year"]) + 1)
    lag1 = c2.number_input("Last year's yield (t/ha)", 0.0, value=float(last["Yield"]), format="%.3f")
    roll3 = c3.number_input("Average of last 3 years (t/ha)", 0.0,
                            value=float(history["Yield"].tail(3).mean()), format="%.3f")
    rain = st.slider("Annual rainfall (mm)", 300.0, 6600.0, float(last["Annual_Rainfall"]), step=10.0)

    names = list(MODELS.values())
    model_name = st.selectbox("Model", names, index=names.index("XGBoost"))
    model_file = [f for f, n in MODELS.items() if n == model_name][0]

    if st.button("Predict", type="primary"):
        row = {"Crop": crop, "State": state, "Season": season, "Crop_Year": int(year),
               "Annual_Rainfall": rain, "Yield_lag1": lag1, "Yield_roll3": roll3,
               "Years_since_last": max(int(year) - int(last["Crop_Year"]), 1)}
        pred = predict_yield(row, model_file)

        c1, c2 = st.columns(2)
        if lag1 > 0:
            change = (pred / lag1 - 1) * 100
            c1.metric("Predicted yield (t/ha)", f"{pred:.3f}", f"{change:+.1f}% vs last year")
        else:
            c1.metric("Predicted yield (t/ha)", f"{pred:.3f}")
        c2.metric("Simple guess (= last year)", f"{lag1:.3f}")
        st.caption("Estimate from state-level data. It doesn't know local soil, irrigation or pests.")

# ---------------- RESULTS ----------------
else:
    st.title("Results on unseen years (2017–2020)")
    comp = load_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))
    st.dataframe(comp, hide_index=True)

    best = comp[~comp["Model"].str.contains("Baseline")].iloc[0]
    base = comp[comp["Model"] == "Last Year (Baseline)"].iloc[0]
    mae_drop = (1 - best["MAE"] / base["MAE"]) * 100
    rmse_drop = (1 - best["RMSE"] / base["RMSE"]) * 100
    st.markdown(f"""
**Best model: {best['Model']}.** Compared with just repeating last year's yield:
- MAE {base['MAE']:.3f} → **{best['MAE']:.3f}** t/ha ({mae_drop:.1f}% lower)
- RMSE {base['RMSE']:.3f} → **{best['RMSE']:.3f}** ({rmse_drop:.1f}% lower, so fewer big misses)
- R² {base['R2']:.3f} → **{best['R2']:.3f}**
- Median % error: baseline {base['MdAPE_%']:.1f}% vs model {best['MdAPE_%']:.1f}%.
  For a normal row last year is already a good guess; the model helps most when yield changes a lot.
""")
    c1, c2 = st.columns(2)
    c1.image(os.path.join(FIGURES_DIR, "model_comparison_bars.png"), caption="Grey = baselines, green = ML models")
    c2.image(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), caption="Best model, test set (log axes)")

    st.subheader("Error by crop (10 most common crops)")
    st.dataframe(load_csv(os.path.join(REPORTS_DIR, "crop_errors.csv")), hide_index=True)

    st.subheader("Which features matter (SHAP)")
    st.caption("The model predicts the change from last year, so this shows what pushes yield up or down. "
               "It is not cause and effect.")
    c1, c2 = st.columns(2)
    c1.image(os.path.join(FIGURES_DIR, "shap_importance_bar.png"))
    c2.dataframe(load_csv(os.path.join(REPORTS_DIR, "shap_feature_importance.csv")), hide_index=True)

    st.subheader("Punjab")
    st.caption("Only 56 Punjab test rows, so these numbers are noisy.")
    st.dataframe(load_csv(os.path.join(REPORTS_DIR, "punjab_metrics.csv")), hide_index=True)
    st.image(os.path.join(FIGURES_DIR, "punjab_historical_trends.png"), caption="Punjab yield over time")
