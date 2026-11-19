# 🌾 AgriVision: Machine Learning-Based Crop Yield Prediction & Agricultural Performance Analysis

An end-to-end Machine Learning system developed to predict agricultural crop yield across Indian states using 24 years of historical data (1997–2020). 

Built with rigorous data leakage prevention, chronological generalization testing, seven regression algorithms, hyperparameter tuning, Explainable AI (SHAP), a dedicated Punjab agronomic case study, and an interactive Streamlit dashboard.

---

## 📑 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Dataset & Data Hygiene](#-dataset--data-hygiene)
3. [Critical Data Leakage Prevention](#-critical-data-leakage-prevention)
4. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
5. [Installation & Setup](#-installation--setup)
6. [Pipeline Execution](#-pipeline-execution)
7. [Model Benchmark & Evaluation Results](#-model-benchmark--evaluation-results)
8. [Explainable AI (SHAP) Insights](#-explainable-ai-shap-insights)
9. [Punjab Regional Agronomic Case Study](#-punjab-regional-agronomic-case-study)
10. [Streamlit Dashboard Guide](#-streamlit-dashboard-guide)
11. [🎓 Comprehensive Viva Voce Preparation Guide](#-comprehensive-viva-voce-preparation-guide)

---

## 🎯 Project Overview & Objectives

Agricultural crop yield prediction is a foundational problem in agronomic economics and food security planning. 

### Core Project Goals:
- **Regression Modeling**: Predict crop yield ($\text{Yield} = \text{Production} / \text{Area}$) based on crop category, state, season, rainfall, input applications, and time horizon.
- **Fair Benchmarking**: Compare 7 regression algorithms (Linear, Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost) under an identical preprocessed feature pipeline.
- **Chronological Generalization**: Train on historical data prior to 2017 (1997–2016, 83.5% of data) and test on subsequent held-out years (2017–2020, 16.5% of data) to simulate real-world forecasting.
- **Model Interpretability**: Apply SHAP (Shapley Additive exPlanations) to explain global feature importance and individual prediction attributions while strictly distinguishing **association from causation**.
- **Regional Deep-Dive**: Provide a focused evaluation on **Punjab**, India's staple grain breadbasket.
- **Interactive UI**: Deliver a multi-page interactive Streamlit dashboard.

---

## 📊 Dataset & Data Hygiene

The project utilizes historical Indian agricultural records (adapted from the Kaggle Indian States Crop Yield Dataset):
- **Total Records**: 19,689 rows
- **Columns**: 10
- **Missing Values**: 0 across all features
- **Duplicate Rows**: 0

### Features:
| Column | Type | Description | Role in Project |
| :--- | :--- | :--- | :--- |
| **`Crop`** | Categorical (55 unique) | Crop cultivated (e.g., Rice, Wheat, Sugarcane, Cotton) | Predictive Feature |
| **`Crop_Year`** | Numerical (1997–2020) | Year of harvest | Predictive Feature / Split |
| **`Season`** | Categorical (6 unique) | Kharif, Rabi, Summer, Autumn, Winter, Whole Year | Predictive Feature |
| **`State`** | Categorical (30 unique) | Indian State / Union Territory | Predictive Feature |
| **`Area`** | Numerical (Hectares) | Cultivated land area | **EXCLUDED (Leakage)** |
| **`Production`** | Numerical (Tonnes) | Total harvest output | **EXCLUDED (Leakage)** |
| **`Annual_Rainfall`** | Numerical (mm) | Total annual precipitation (301 mm – 6,552 mm) | Predictive Feature |
| **`Fertilizer`** | Numerical (kg) | Total fertilizer applied | Predictive Feature |
| **`Pesticide`** | Numerical (kg) | Total pesticide applied | Predictive Feature |
| **`Yield`** | Numerical (Float) | Crop yield per unit land area | **Prediction Target ($y$)** |

### Data Hygiene Step:
All 19,689 rows in `Season` contained trailing whitespace (e.g. `'Kharif     '`), and 172 rows in `Crop` contained trailing spaces (e.g. `'Coconut '`). These were stripped to ensure clean categorical encoding and robust dropdown selection in the UI.

---

## ⚠️ Critical Data Leakage Prevention

In academic machine learning, data leakage often occurs silently. Our project explicitly investigates and resolves two major sources of leakage:

1. **Target Derivation Leakage (`Production` & `Area`)**:
   - $\text{Yield} = \frac{\text{Production}}{\text{Area}}$.
   - Our data inspection confirmed that the median ratio of $\text{Yield} / (\text{Production} / \text{Area})$ is **0.996** (virtually exact).
   - If `Production` or `Area` is passed into the model, any regression algorithm simply learns arithmetic division ($R^2 \approx 1.0$), bypassing all biological and climatic relationships.
   - Total `Production` is an **outcome variable** observed *after* harvest, meaning it is impossible to know beforehand.
   - **Decision**: Both `Production` and `Area` are **completely excluded** from the feature set.

2. **Temporal Leakage (Random vs Chronological Split)**:
   - Random $k$-fold cross-validation interpolates past yields from future knowledge.
   - **Decision**: We implement a strict **chronological split**:
     - **Training Set**: 1997–2016 (16,440 rows, 83.5%)
     - **Held-Out Test Set**: 2017–2020 (3,249 rows, 16.5%)
     - No unseen crops or states exist in the test set.

---

## 🏗️ Project Architecture & Directory Structure

```
AgriVision/
│
├── data/
│   ├── raw/
│   │   └── crop_yield.csv              # Raw downloaded dataset (19,689 rows)
│   └── processed/
│       ├── cleaned_crop_yield.csv      # Trimmed, cleaned dataset
│       ├── train_data.csv              # Training set (1997-2016)
│       ├── test_data.csv               # Held-out test set (2017-2020)
│       ├── X_train_proc.npy            # One-hot encoded & scaled train features
│       ├── X_test_proc.npy             # Transformed test features
│       ├── y_train.npy                 # Train targets
│       ├── y_test.npy                  # Test targets
│       └── metadata.json               # Feature names, categories & bounds
│
├── notebooks/
│   ├── exploratory_analysis.ipynb      # Step-by-step Jupyter analysis notebook
│   └── create_notebook.py              # Notebook generator script
│
├── src/
│   ├── preprocessing.py               # Cleaning, leakage checks, split & encoding
│   ├── train_models.py                 # 7 algorithms & TimeSeriesSplit tuning
│   ├── evaluate_models.py              # Test evaluation, metrics & figure generation
│   ├── explainability.py               # SHAP TreeExplainer & feature ranking
│   └── prediction.py                   # Inference engine with bounds & warnings
│
├── models/
│   ├── preprocessor.joblib             # Fitted ColumnTransformer (OHE + Scaler)
│   ├── linear_regression.joblib        # Baseline model
│   ├── ridge_regression.joblib         # L2 regularized model
│   ├── random_forest.joblib            # Tuned Random Forest
│   ├── extra_trees.joblib              # Extremely Randomized Trees (Best)
│   ├── gradient_boosting.joblib        # Gradient Boosting Regressor
│   ├── xgboost.joblib                  # Tuned XGBoost
│   └── catboost.joblib                 # Tuned CatBoost
│
├── reports/
│   ├── figures/
│   │   ├── model_comparison_bars.png   # MAE, RMSE, R² comparison chart
│   │   ├── actual_vs_predicted.png     # Log-log scatter plot of predictions
│   │   ├── residuals_distribution.png  # Error histogram & residual plot
│   │   ├── shap_summary.png            # SHAP beeswarm summary plot
│   │   ├── shap_importance_bar.png     # Global feature importance rankings
│   │   ├── punjab_historical_trends.png# Yield trends over time in Punjab
│   │   └── punjab_crop_yield_comparison.png # Boxplots of top Punjab crops
│   ├── model_comparison.csv            # Comprehensive test evaluation table
│   ├── punjab_metrics.csv              # Punjab-specific test evaluation table
│   ├── hyperparameter_tuning.json      # Best parameters from RandomizedSearchCV
│   └── shap_feature_importance.json    # SHAP rankings with agronomic descriptions
│
├── app/
│   └── app.py                          # 6-page interactive Streamlit dashboard
│
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation & Viva preparation guide
└── .gitignore                          # Ignored caches & temp files
```

---

## ⚙️ Installation & Setup

### 1. Clone or Navigate to the Project
```bash
cd C:\Users\12sin\.gemini\antigravity\scratch\AgriVision
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Execution

The pipeline is completely reproducible and can be executed via terminal commands:

### Step 1: Data Preprocessing
Cleans strings, excludes leaky features, performs chronological split, fits ColumnTransformer:
```bash
python src/preprocessing.py
```

### Step 2: Model Training & Hyperparameter Tuning
Trains all 7 models, tunes Random Forest, XGBoost, and CatBoost with `TimeSeriesSplit`:
```bash
python src/train_models.py
```

### Step 3: Model Evaluation & Visualization
Computes MAE, RMSE, $R^2$ on held-out test data and Punjab subset; generates figures:
```bash
python src/evaluate_models.py
```

### Step 4: Explainable AI with SHAP
Computes TreeExplainer SHAP values and outputs feature attribution plots:
```bash
python src/explainability.py
```

### Step 5: Launch the Streamlit Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🏆 Model Benchmark & Evaluation Results

All models were evaluated on the held-out test set (2017–2020, 3,249 unseen records):

### Overall Held-Out Test Evaluation:
| Rank | Model | MAE | RMSE | $R^2$ Score | Key Characteristics |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **Extra Trees Regressor** | **8.78** | **123.40** | **0.9787** | Extremely randomized thresholds; best variance reduction |
| 🥈 | **Random Forest Regressor** | **9.77** | **138.28** | **0.9732** | Strong bagging ensemble; robust to outliers |
| 🥉 | **CatBoost Regressor** | **12.73** | **144.38** | **0.9708** | Symmetric oblivious trees with L2 leaf regularization |
| 4 | **Gradient Boosting Regressor** | **11.27** | **151.18** | **0.9680** | Sequential residual learning |
| 5 | **XGBoost Regressor** | **15.67** | **165.24** | **0.9617** | Second-order gradient boosting with tree pruning |
| 6 | **Linear Regression (Baseline)** | **56.55** | **309.03** | **0.8662** | Simple parametric baseline; assumes linear relationships |
| 7 | **Ridge Regression ($L_2$)** | **59.68** | **318.19** | **0.8581** | Linear model with ridge penalty ($\alpha=10.0$) |

### Punjab Subset Held-Out Test Evaluation (58 records):
| Rank | Model | Punjab MAE | Punjab RMSE | Punjab $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 | **Random Forest Regressor** | **0.6043** | **1.7361** | **0.9906** |
| 🥈 | **Extra Trees Regressor** | **0.7729** | **1.9893** | **0.9876** |
| 🥉 | **Gradient Boosting** | **1.9274** | **4.8869** | **0.9253** |
| 4 | **XGBoost Regressor** | **3.6678** | **8.0346** | **0.7982** |
| 5 | **CatBoost Regressor** | **3.8512** | **8.6655** | **0.7652** |
| 6 | **Ridge Regression** | **12.6813** | **14.2550** | **0.3646** |
| 7 | **Linear Regression** | **13.9789** | **15.0446** | **0.2923** |

---

## 🔍 Explainable AI (SHAP) Insights

SHAP (Shapley Additive exPlanations) grounds feature importance in cooperative game theory:

### Top Influential Features:
1. **`Crop_Coconut`** (Mean |SHAP| = 109.55): Coconuts yield thousands of nuts per hectare, driving the gross magnitude of the numerical yield scale.
2. **`Annual_Rainfall`** (Mean |SHAP| = 8.15): Primary hydrological driver for rainfed and semi-irrigated zones.
3. **`State_Karnataka` & `State_West Bengal`** (Mean |SHAP| = 4.09 & 3.64): Captures state-level agro-ecological suitability and irrigation networks.
4. **`Crop_Year`** (Mean |SHAP| = 3.26): Proxy for multi-decade technological advancement (high-yield hybrid seeds, mechanization).
5. **`Season_Whole Year` & `Crop_Sugarcane`** (Mean |SHAP| = 2.78 & 2.73): Distinguishes perennial/annual crops from seasonal field crops.

> ⚠️ **Association vs. Causation**: SHAP highlights statistical associations within historical records. It does **not** imply that artificially increasing pesticide or fertilizer will causally increase crop yield without suitable water, soil health, and pest pressure.

---

## 🌾 Punjab Regional Agronomic Case Study

- **Dataset Representation**: 397 historical records across 23 crops from 1997 to 2019.
- **Top Crops**: Wheat, Rice, Cotton(lint), Maize, Sugarcane, Gram.
- **Why District-Level Predictions Are Not Possible**: The dataset is aggregated at the state level.
- **Generalization**: On the 58 unseen Punjab test records, Random Forest achieved an MAE of only **0.60 tonnes/ha** and $R^2 = 0.9906$. Tree models successfully capture Punjab's intensive irrigation and high agricultural productivity.

---

## 🖥️ Streamlit Dashboard Guide

The web dashboard is organized into 6 dedicated pages:
1. **Home**: High-level metrics, system architecture diagram, and project objectives.
2. **Dataset Exploration**: Interactive data table, crop/season distributions, correlation heatmap, and data leakage documentation.
3. **Yield Prediction**: Form with sliders/dropdowns, input validation, extrapolation warnings, model selector, and prediction disclaimer.
4. **Model Comparison**: Full comparative benchmark table, bar charts, actual vs predicted scatter plot (log scale), and residual distribution.
5. **Explainable AI (SHAP)**: Global beeswarm summary plot, feature ranking table, and association vs causation guidance.
6. **Punjab Analysis**: Punjab crop yield trajectories over time, crop variability boxplots, and regional evaluation metrics.

---

## 🎓 Comprehensive Viva Voce Preparation Guide

### 1. Algorithms Explained in Plain Language

#### A. Linear Regression (Baseline)
- **Concept**: Fits a hyper-plane by minimizing Ordinary Least Squares (OLS) residual sum of squares: $\min \sum (y_i - \hat{y}_i)^2$.
- **Role**: Serves as our baseline to test whether a simple linear combination of inputs can forecast yield.
- **Limitations**: Fails to capture non-linear agro-climatic saturation points (e.g., excessive rainfall causing waterlogging).

#### B. Ridge Regression ($L_2$ Regularization)
- **Concept**: Adds an $L_2$ penalty to the loss function: $\text{Loss} = \text{OLS} + \alpha \sum w_j^2$.
- **Role**: Shrinks regression coefficients towards zero, mitigating multicollinearity between one-hot encoded categories and correlated weather inputs.

#### C. Random Forest Regressor (Bagging)
- **Concept**: An ensemble of independent Decision Trees trained on bootstrap samples (Bagging) with random feature subsets at each split. Final prediction is the average of all tree predictions.
- **Why it performs well**: Reduces variance without increasing bias, naturally capturing non-linear interactions between crop type, season, and rainfall.

#### D. Extra Trees Regressor (Extremely Randomized Trees)
- **Concept**: Similar to Random Forest, but selects cut-points completely at random for each feature candidate rather than calculating the optimal split threshold.
- **Why it won overall**: Random thresholds add further decorrelation between trees, yielding lower variance and superior generalization ($R^2 = 0.9787$).

#### E. Gradient Boosting Regressor (GBR)
- **Concept**: An ensemble where shallow decision trees are trained sequentially. Each subsequent tree fits to the pseudo-residuals (negative gradient of loss) of the previous trees.
- **Key Parameters**: `learning_rate`, `n_estimators`, `max_depth`.

#### F. XGBoost Regressor (Extreme Gradient Boosting)
- **Concept**: Advanced gradient boosting utilizing second-order Taylor expansion (gradients and Hessians), built-in $L_1/L_2$ regularization, and column subsampling.
- **Advantage**: Fast execution, handles sparse one-hot matrices efficiently.

#### G. CatBoost Regressor
- **Concept**: Gradient boosting utilizing symmetric (oblivious) decision trees, reducing overfitting and providing consistent prediction times.

---

### 2. Evaluation Metrics Explained

- **MAE (Mean Absolute Error)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
  Measures average magnitude of error in the exact same units as the target (tonnes/ha). Robust to extreme outliers.
- **RMSE (Root Mean Squared Error)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
  Penalizes large prediction errors more severely than MAE due to squaring.
- **$R^2$ Score (Coefficient of Determination)**:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  Proportion of target variance explained by the model relative to a naive mean baseline ($1.0$ = perfect prediction, $0.0$ = baseline mean).

---

### 3. Top Viva Questions & Model Answers

**Q1: Why did you not include `Production` and `Area` as input features?**
> *"Because Yield is mathematically calculated as Production divided by Area. Including either or both creates direct target leakage: the model would trivially learn arithmetic division rather than learning agricultural dependencies. Additionally, total harvest production is only observed after harvest, making it unavailable for pre-harvest forecasting."*

**Q2: Why did you use a chronological split instead of a random 80/20 train/test split?**
> *"Agricultural forecasting is inherently temporal. A random split causes temporal leakage where future records inform past predictions. By training strictly on years prior to 2017 and testing on 2017–2020, we evaluated genuine forward forecasting ability."*

**Q3: Why does Extra Trees or Random Forest outperform Linear Regression so dramatically?**
> *"Agricultural yield depends on sharp non-linear thresholds and complex categorical interactions (e.g. Rice requires high water during Kharif, while Wheat requires moderate water during Rabi). Tree ensembles easily create rectangular decision boundaries for these interactions, whereas linear models assume constant additive slopes."*

**Q4: Does SHAP prove that adding more fertilizer will increase yield?**
> *"No. SHAP identifies feature attribution within the trained model (association), not causal reality. High input usage correlates historically with productive farms, but applying excess fertilizer in drought conditions will damage crops rather than increase yield."*

**Q5: Can this system predict crop yield at the district or farm field level?**
> *"No. The underlying dataset is aggregated at the state level. Predicting at the district or field level would require localized district-level weather, soil testing data (NPK, pH), and field-level satellite indices (NDVI)."*
