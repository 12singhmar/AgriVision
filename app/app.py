"""
AgriVision - Streamlit app
==========================
Three pages:
1. Data     - what the dataset looks like and what was cleaned
2. Predict  - forecast next year's yield for a Crop/State/Season
3. Results  - model vs baseline comparison, feature importance, Punjab
Run with:  streamlit run app/app.py
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from prediction import AgriVisionPredictor  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

st.set_page_config(page_title="AgriVision", page_icon="🌾", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"))
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        meta = json.load(f)
    reports = {name: pd.read_csv(os.path.join(REPORTS_DIR, f"{name}.csv"))
               for name in ["model_comparison", "punjab_metrics", "crop_errors", "shap_feature_importance"]}
    return df, meta, reports


@st.cache_resource
def load_predictor():
    return AgriVisionPredictor()


def show_figure(name, caption):
    path = os.path.join(FIGURES_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")


df, meta, reports = load_data()
predictor = load_predictor()

st.sidebar.title("🌾 AgriVision")
page = st.sidebar.radio("Page", ["Data", "Predict", "Results"])
st.sidebar.caption("Crop yield forecasting for Indian states, 1997–2020")

# ---------------------------------------------------------------- DATA
if page == "Data":
    st.title("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows used", f"{len(df):,}")
    c2.metric("Crops", df["Crop"].nunique())
    c3.metric("States", df["State"].nunique())
    c4.metric("Years", f"{df['Crop_Year'].min()}–{df['Crop_Year'].max()}")

    st.markdown("""
**Source:** Kaggle "Crop Yield in Indian States" (one row per crop × season × state × year).

**Cleaning and decisions**
- Stripped trailing spaces from `Season` and `Crop`.
- Removed **Coconut**: its yield is in nuts/ha (thousands), every other crop is in tonnes/ha.
- **Production** and **Area** are not used: Yield = Production / Area, and Production is only known after harvest.
- **Fertilizer** and **Pesticide** are not used: in this dataset they equal Area × a fixed rate per year, so they are not real measurements.
- Added **lag features** from each crop/state/season's own past: last year's yield, 3-year average and years since the last record.
  The first year of each series has no history and was dropped.
- Train on years before 2017, test on 2017–2020.
""")
    st.subheader("Sample rows")
    st.dataframe(df.head(50), width="stretch")
    st.subheader("Most common crops")
    st.bar_chart(df["Crop"].value_counts().head(15))

# ---------------------------------------------------------------- PREDICT
elif page == "Predict":
    st.title("Forecast next year's yield")
    cats = meta["categories"]
    c1, c2, c3 = st.columns(3)
    crop = c1.selectbox("Crop", cats["crops"], index=cats["crops"].index("Wheat"))
    state = c2.selectbox("State", cats["states"], index=cats["states"].index("Punjab"))
    season = c3.selectbox("Season", cats["seasons"], index=cats["seasons"].index("Rabi"))

    history = df[(df["Crop"] == crop) & (df["State"] == state) & (df["Season"] == season)].sort_values("Crop_Year")
    if history.empty:
        st.warning("No history for this combination, so last year's yield is unknown. Pick another combination.")
        st.stop()

    last = history.iloc[-1]
    st.caption(f"History found: {len(history)} years, latest {int(last['Crop_Year'])} "
               f"with yield {last['Yield']:.2f} t/ha. Values below are pre-filled from it; you can change them.")
    st.line_chart(history.set_index("Crop_Year")["Yield"])

    c1, c2, c3 = st.columns(3)
    year = c1.number_input("Year to forecast", 1998, 2030, int(last["Crop_Year"]) + 1)
    lag1 = c2.number_input("Last year's yield (t/ha)", 0.0, value=float(last["Yield"]), format="%.3f")
    roll3 = c3.number_input("Average of last 3 years (t/ha)", 0.0,
                            value=float(history["Yield"].tail(3).mean()), format="%.3f")
    rain = st.slider("Annual rainfall (mm)", 300.0, 6600.0, float(last["Annual_Rainfall"]), step=10.0)
    models = {"XGBoost": "xgboost", "Random Forest": "random_forest", "Gradient Boosting": "gradient_boosting",
              "Ridge Regression": "ridge_regression", "Linear Regression": "linear_regression"}
    model_name = st.selectbox("Model", list(models))

    if st.button("Predict", type="primary"):
        row = {"Crop": crop, "State": state, "Season": season, "Crop_Year": int(year),
               "Annual_Rainfall": rain, "Yield_lag1": lag1, "Yield_roll3": roll3, "Years_since_last": 1}
        out = predictor.predict(row, models[model_name])
        c1, c2 = st.columns(2)
        c1.metric("Predicted yield (t/ha)", f"{out['predicted_yield']:.3f}",
                  f"{out['change_pct']:+.1f}% vs last year" if out["change_pct"] is not None else None)
        c2.metric("Naive guess (= last year)", f"{lag1:.3f}")
        for w in out["warnings"]:
            st.warning(w)
        st.caption("Statistical estimate from state-level historical data; it does not know local soil, irrigation or pests. "
                   "Tree models treat any year after 2016 like 2016; the trend comes from the lag inputs instead.")

# ---------------------------------------------------------------- RESULTS
else:
    st.title("Results on unseen years (2017–2020)")
    comp = reports["model_comparison"]
    st.dataframe(comp, width="stretch", hide_index=True)

    ml = comp[~comp["Model"].str.contains("Baseline")].iloc[0]
    base = comp[comp["Model"] == "Last Year (Baseline)"].iloc[0]
    st.markdown(f"""
**Best model: {ml['Model']}.** Compared with simply repeating last year's yield:
- MAE {base['MAE']:.3f} → **{ml['MAE']:.3f}** t/ha ({100 * (1 - ml['MAE'] / base['MAE']):.1f}% lower)
- RMSE {base['RMSE']:.3f} → **{ml['RMSE']:.3f}** ({100 * (1 - ml['RMSE'] / base['RMSE']):.1f}% lower, so fewer large misses)
- R² {base['R2']:.3f} → **{ml['R2']:.3f}**
- Median % error: baseline {base['MdAPE_%']:.1f}% vs model {ml['MdAPE_%']:.1f}%. For a typical row, last year's yield is already a very good guess;
  the model's gain comes mostly from the rows where yield changed a lot.
""")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("model_comparison_bars.png", "Grey = baselines, green = ML models")
    with c2:
        show_figure("actual_vs_predicted.png", "Best model, test set (log axes)")

    st.subheader("Error by crop (10 most common crops)")
    st.dataframe(reports["crop_errors"], width="stretch", hide_index=True)

    st.subheader("Which features matter (SHAP)")
    st.caption("Models predict the change from last year, so this shows what pushes yield above or below last year's. "
               "It shows association inside the model, not cause and effect.")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("shap_importance_bar.png", "Mean absolute SHAP value")
    with c2:
        st.dataframe(reports["shap_feature_importance"], width="stretch", hide_index=True)

    st.subheader("Punjab")
    n_pj = int(((df["State"] == "Punjab") & (df["Crop_Year"] >= meta["split_year"])).sum())
    st.caption(f"Only {n_pj} Punjab test rows, so these numbers are noisy.")
    st.dataframe(reports["punjab_metrics"], width="stretch", hide_index=True)
    show_figure("punjab_historical_trends.png", "Punjab yield over time")
