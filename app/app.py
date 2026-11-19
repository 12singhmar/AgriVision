"""
AgriVision - Streamlit Web Application
=======================================
Interactive, modern agricultural intelligence dashboard for crop yield prediction,
model benchmarking, explainable AI (SHAP), and focused Punjab regional analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="AgriVision | ML Crop Yield Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "crop_yield.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b6b50;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .info-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #c8e6c9;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #ffe082;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    """Loads cleaned dataset, metadata, and evaluation reports."""
    cleaned_df = pd.read_csv(os.path.join(DATA_DIR, "cleaned_crop_yield.csv"))
    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    comp_df = pd.read_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))
    punjab_metrics_df = pd.read_csv(os.path.join(REPORTS_DIR, "punjab_metrics.csv"))
    
    with open(os.path.join(REPORTS_DIR, "shap_feature_importance.json"), "r") as f:
        shap_rankings = json.load(f)

    return cleaned_df, metadata, comp_df, punjab_metrics_df, shap_rankings


@st.cache_resource
def load_predictor():
    """Imports predictor module and loads trained preprocessor."""
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    from prediction import AgriVisionPredictor
    return AgriVisionPredictor()


# Load datasets
try:
    df, metadata, comp_df, punjab_metrics_df, shap_rankings = load_app_data()
    predictor = load_predictor()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading project artifacts: {e}. Please ensure preprocessing and training scripts have run.")
    data_loaded = False


# Sidebar Navigation
st.sidebar.title("🌾 AgriVision")
st.sidebar.caption("Machine Learning Agricultural Performance & Yield Prediction")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Home",
        "Dataset Exploration",
        "Yield Prediction",
        "Model Comparison",
        "Explainable AI (SHAP)",
        "Punjab Analysis",
    ],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset Source**: Kaggle Indian Crop Yield Dataset (1997–2020)\n\n"
    "**Models Evaluated**: 7 Regression Algorithms\n\n"
    "**Validation Scheme**: Chronological Held-Out Test Set"
)


# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.markdown('<div class="main-title">🌾 AgriVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Machine Learning-Based Crop Yield Prediction and Agricultural Performance Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h4>Total Records</h4><h2>19,689</h2><p>Historical Indian Farm Records</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Crop Varieties</h4><h2>55</h2><p>Cereals, Pulses, Cash Crops</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>States Covered</h4><h2>30</h2><p>States & Union Territories</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h4>Time Horizon</h4><h2>1997–2020</h2><p>24 Years of Agronomic Data</p></div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Project Overview")
    st.markdown("""
    **AgriVision** is an end-to-end Machine Learning system developed to forecast agricultural crop yield across Indian states 
    using historical agronomic, climatic, and seasonal indicators. 

    The system focuses on rigorous engineering practices:
    - **Data Leakage Elimination**: Investigating and removing direct mathematical dependencies between *Production*, *Cultivated Area*, and *Yield*.
    - **Chronological Generalization**: Training strictly on earlier years (1997–2016) and evaluating on subsequent unseen years (2017–2020).
    - **Multi-Algorithm Benchmarking**: Implementing, tuning, and comparing 7 distinct regression approaches from classical linear baselines to state-of-the-art tree ensembles.
    - **Explainable AI (XAI)**: Applying **SHAP (Shapley Additive exPlanations)** to uncover model decision pathways while maintaining the critical distinction between correlation and causation.
    - **Focused Regional Analytics**: Dedicated performance evaluation and crop-yield trend analysis for the agricultural heartland of **Punjab**.
    """)

    st.markdown("### 🔬 System Architecture & Workflow")
    st.markdown("""
    ```
    Raw Historical Dataset (19,689 Records)
             │
             ▼
    Data Preprocessing & Hygiene (Whitespace Trimming, Leakage Exclusion of Area & Production)
             │
             ▼
    Chronological Split (Train: 1997–2016 [83.5%] | Test: 2017–2020 [16.5%])
             │
             ▼
    Feature Transformation (StandardScaler for Weather/Inputs + OneHotEncoder for Categoricals)
             │
             ▼
    7 Regression Algorithms Benchmark (Linear, Ridge, RF, Extra Trees, GBR, XGBoost, CatBoost)
             │
             ▼
    TimeSeriesSplit Hyperparameter Tuning (Random Forest, XGBoost, CatBoost)
             │
             ▼
    Explainable AI with SHAP + Focused Punjab Agronomic Analysis + Interactive UI
    ```
    """)


# ==========================================
# PAGE 2: DATASET EXPLORATION
# ==========================================
elif page == "Dataset Exploration":
    st.header("📊 Dataset Exploration & Data Hygiene")
    st.markdown("Explore the underlying distribution, categorical diversity, and statistical integrity of the historical dataset.")

    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Preview", "Distributions & Heatmap", "Data Leakage Investigation", "Summary Statistics"])

    with tab1:
        st.subheader("Historical Dataset Sample")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"Showing sample from {len(df):,} total records and {df.shape[1]} features.")

    with tab2:
        st.subheader("Feature Distributions & Correlation Analysis")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Top 15 Most Frequently Cultivated Crops**")
            top_crops = df["Crop"].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=top_crops.values, y=top_crops.index, ax=ax, palette="mako")
            ax.set_xlabel("Record Count")
            plt.tight_layout()
            st.pyplot(fig)

        with colB:
            st.markdown("**Season Distribution**")
            season_counts = df["Season"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.pie(season_counts, labels=season_counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("**Correlation Heatmap of Numerical Features**")
        fig, ax = plt.subplots(figsize=(9, 5))
        num_cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax, cbar=True)
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("⚠️ Critical Data Leakage Prevention Analysis")
        st.markdown("""
        In agricultural machine learning, target leakage occurs when an input feature mathematically reveals or is derived 
        from the prediction target.
        
        **1. Why Production & Area were EXCLUDED from the Predictive Features:**
        - Mathematically, $\\text{Yield} = \\frac{\\text{Production}}{\\text{Area}}$.
        - As demonstrated in our data hygiene check, the median ratio of $\\text{Yield} / (\\text{Production} / \\text{Area})$ is **0.996** (virtually 1.0).
        - If `Production` or `Area` were retained as inputs, any machine learning algorithm would simply learn a trivial arithmetic division rather than learning agricultural dynamics.
        - Furthermore, total harvest `Production` is only known *after* harvest, making it completely unavailable at sowing or growing time!

        **2. What about Fertilizer and Pesticide?**
        - In the raw dataset, `Fertilizer` and `Pesticide` represent total kilograms across the entire state crop acreage ($r = 0.973$ with Area).
        - They are retained in the feature set as proxies for regional input intensity, but models are trained without `Area` to ensure no backdoor area leakage occurs.
        """)

    with tab4:
        st.subheader("Statistical Summary of Numerical Features")
        st.dataframe(df.describe().T, use_container_width=True)


# ==========================================
# PAGE 3: YIELD PREDICTION
# ==========================================
elif page == "Yield Prediction":
    st.header("🌱 Live Crop Yield Prediction")
    st.markdown("Enter crop cultivation, weather, and input parameters to forecast estimated yield per hectare.")

    col1, col2 = st.columns([1, 1])

    categories = metadata["categories"]
    ranges = metadata["numerical_ranges"]

    with col1:
        st.subheader("🌾 Cultivation Context")
        crop_select = st.selectbox("Select Crop", categories["crops"], index=categories["crops"].index("Wheat") if "Wheat" in categories["crops"] else 0)
        state_select = st.selectbox("Select State", categories["states"], index=categories["states"].index("Punjab") if "Punjab" in categories["states"] else 0)
        season_select = st.selectbox("Select Season", categories["seasons"], index=categories["seasons"].index("Rabi") if "Rabi" in categories["seasons"] else 0)
        year_input = st.number_input("Crop Year", min_value=1997, max_value=2030, value=2024, step=1)

    with col2:
        st.subheader("🌦️ Environmental & Input Conditions")
        rainfall_input = st.slider(
            "Annual Rainfall (mm)",
            min_value=float(ranges["Annual_Rainfall"]["min"]),
            max_value=float(ranges["Annual_Rainfall"]["max"]),
            value=float(ranges["Annual_Rainfall"]["median"]),
            step=25.0,
            help="Historical regional rainfall range: 301 mm to 6,552 mm"
        )
        fertilizer_input = st.number_input(
            "Total Fertilizer Usage (kg)",
            min_value=0.0,
            max_value=float(ranges["Fertilizer"]["max"]),
            value=float(ranges["Fertilizer"]["median"]),
            step=50000.0,
            help=f"Dataset median: {ranges['Fertilizer']['median']:,.0f} kg"
        )
        pesticide_input = st.number_input(
            "Total Pesticide Usage (kg)",
            min_value=0.0,
            max_value=float(ranges["Pesticide"]["max"]),
            value=float(ranges["Pesticide"]["median"]),
            step=100.0,
            help=f"Dataset median: {ranges['Pesticide']['median']:,.0f} kg"
        )

        model_options = {
            "Extra Trees (Best Overall)": "extra_trees",
            "Random Forest": "random_forest",
            "CatBoost Regressor": "catboost",
            "Gradient Boosting": "gradient_boosting",
            "XGBoost Regressor": "xgboost",
            "Linear Regression (Baseline)": "linear_regression",
            "Ridge Regression": "ridge_regression",
        }
        selected_model_label = st.selectbox("Select Model for Prediction", list(model_options.keys()), index=0)
        chosen_model_slug = model_options[selected_model_label]

    predict_btn = st.button("🚀 Predict Crop Yield", type="primary", use_container_width=True)

    if predict_btn:
        input_data = {
            "Crop": crop_select,
            "State": state_select,
            "Season": season_select,
            "Crop_Year": year_input,
            "Annual_Rainfall": rainfall_input,
            "Fertilizer": fertilizer_input,
            "Pesticide": pesticide_input,
        }

        try:
            prediction_output = predictor.predict_single(input_data, model_slug=chosen_model_slug)

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Estimated Crop Yield</h4>
                    <h1 style="color: #2e7d32; font-size: 3rem;">{prediction_output['predicted_yield']:.2f}</h1>
                    <p><strong>Units:</strong> Metric Tonnes / Hectare (or Nuts/Ha for Coconut)</p>
                    <p><strong>Model:</strong> {prediction_output['model_used']}</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                if prediction_output["has_warnings"]:
                    for w in prediction_output["warnings"]:
                        st.warning(w)
                else:
                    st.success("✅ Inputs are safely within historical training distribution bounds.")

                st.info(f"💡 **Model Estimate Context**: {prediction_output['disclaimer']}")

        except Exception as err:
            st.error(f"Prediction failed: {err}")


# ==========================================
# PAGE 4: MODEL COMPARISON
# ==========================================
elif page == "Model Comparison":
    st.header("🏆 Regression Algorithm Benchmark & Evaluation")
    st.markdown("Comprehensive performance evaluation of all 7 regression models on the **held-out test set (2017–2020)**.")

    st.subheader("📋 Model Performance Comparison Table")
    st.dataframe(
        comp_df.style.highlight_max(subset=["R2_Score"], color="#c8e6c9")
                     .highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9")
                     .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2_Score": "{:.4f}"}),
        use_container_width=True
    )

    st.markdown("### 📊 Metric Visualizations")
    bar_img_path = os.path.join(FIGURES_DIR, "model_comparison_bars.png")
    if os.path.exists(bar_img_path):
        st.image(bar_img_path, caption="Comparative Performance: R² Score, MAE, and RMSE across all 7 Models", use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        act_pred_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
        if os.path.exists(act_pred_path):
            st.image(act_pred_path, caption="Actual vs. Predicted Yield on Held-Out Test Set (Log Scale)", use_container_width=True)

    with colB:
        resid_path = os.path.join(FIGURES_DIR, "residuals_distribution.png")
        if os.path.exists(resid_path):
            st.image(resid_path, caption="Residual (Prediction Error) Distribution", use_container_width=True)

    st.markdown("### 🎓 Key Performance Insights for Viva")
    st.markdown("""
    1. **Tree Ensembles Outperform Linear Baselines**: 
       - Linear Regression ($R^2 \\approx 0.866$) and Ridge ($R^2 \\approx 0.858$) perform moderately, but suffer higher MAE (~56.5) because agricultural yield exhibits strong non-linear thresholds (e.g. rainfall saturation, crop-specific biological limits).
       - **Extra Trees Regressor** ($R^2 = 0.9787$, $\\text{MAE} = 8.78$) and **Random Forest Regressor** ($R^2 = 0.9732$, $\\text{MAE} = 9.77$) achieve top ranks by effectively partitioning categorical interactions (Crop $\\times$ Season $\\times$ State).
    2. **Boosting Performance**:
       - Gradient Boosting, CatBoost, and XGBoost all achieve strong $R^2 > 0.96$. CatBoost performs especially well with one-hot transformed features and L2 regularization.
    3. **Generalization Over Time**:
       - Testing exclusively on subsequent years (2017–2020) guarantees that the reported scores reflect genuine future forecast capability, not data interpolation.
    """)


# ==========================================
# PAGE 5: EXPLAINABLE AI (SHAP)
# ==========================================
elif page == "Explainable AI (SHAP)":
    st.header("🔍 Explainable AI (XAI) with SHAP")
    st.markdown("Uncovering model decision pathways and feature contributions using Shapley Additive exPlanations.")

    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Crucial Viva Distinction: Association vs. Causation</strong><br>
        A high SHAP value indicates that a feature strongly influences the model's statistical prediction. 
        It does <em>not</em> prove that artificially increasing that feature will causally increase real-world crop yield. 
        For example, high pesticide application is strongly associated with high-yield cash crops, but spraying excess pesticide on a dry grain field will not increase yield!
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        shap_bar_path = os.path.join(FIGURES_DIR, "shap_importance_bar.png")
        if os.path.exists(shap_bar_path):
            st.image(shap_bar_path, caption="Global Feature Importance (Mean Absolute SHAP Value)", use_container_width=True)

    with colB:
        shap_sum_path = os.path.join(FIGURES_DIR, "shap_summary.png")
        if os.path.exists(shap_sum_path):
            st.image(shap_sum_path, caption="SHAP Beeswarm Summary Plot (Feature Impact & Direction)", use_container_width=True)

    st.subheader("📋 Top Influential Features & Agronomic Interpretation")
    ranking_df = pd.DataFrame(shap_rankings)
    st.dataframe(ranking_df, use_container_width=True)


# ==========================================
# PAGE 6: PUNJAB ANALYSIS
# ==========================================
elif page == "Punjab Analysis":
    st.header("🌾 Focused Agronomic Analysis: Punjab")
    st.markdown("Detailed investigation of crop productivity, historical trends, and model performance in the agricultural powerhouse of Punjab.")

    punjab_df = df[df["State"] == "Punjab"].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Punjab Records</h4><h2>{len(punjab_df)}</h2><p>Historical Records (1997–2019)</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Crops Grown</h4><h2>{punjab_df["Crop"].nunique()}</h2><p>Major: Wheat, Rice, Cotton, Maize</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h4>Mean Wheat Yield</h4><h2>{punjab_df[punjab_df["Crop"]=="Wheat"]["Yield"].mean():.2f}</h2><p>Tonnes / Hectare (Consistently High)</p></div>', unsafe_allow_html=True)

    st.markdown("### 📈 Historical Crop Yield Trends in Punjab")
    trend_path = os.path.join(FIGURES_DIR, "punjab_historical_trends.png")
    if os.path.exists(trend_path):
        st.image(trend_path, caption="Punjab Crop Yield Trajectory Over Time (1997–2019)", use_container_width=True)

    box_path = os.path.join(FIGURES_DIR, "punjab_crop_yield_comparison.png")
    if os.path.exists(box_path):
        st.image(box_path, caption="Punjab Crop Yield Variability Distribution", use_container_width=True)

    st.subheader("🎯 Model Performance on Punjab Held-Out Test Set (58 Records)")
    st.markdown("""
    Because the Kaggle dataset is recorded at the state level (not district level), we evaluate how accurately 
    our India-wide models generalize to the Punjab regional subset.
    """)
    st.dataframe(
        punjab_metrics_df.style.highlight_max(subset=["Punjab_R2_Score"], color="#c8e6c9")
                               .highlight_min(subset=["Punjab_MAE", "Punjab_RMSE"], color="#c8e6c9")
                               .format({"Punjab_MAE": "{:.4f}", "Punjab_RMSE": "{:.4f}", "Punjab_R2_Score": "{:.4f}"}),
        use_container_width=True
    )
    st.success("🌟 **Outstanding Result**: Random Forest and Extra Trees achieved an MAE of ~0.60 to 0.77 tonnes/ha and R² > 0.98 on Punjab test records!")
