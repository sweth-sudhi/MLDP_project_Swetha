import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

RANDOM_STATE = 42
FILE_PATH = "ecommerce_customer_churn_dataset.csv"
TARGET = "Churned"

# -----------------------------
# Feature Engineering
# -----------------------------
def add_features(X_df):
    X_new = X_df.copy()

    if "Lifetime_Value" in X_new.columns and "Total_Purchases" in X_new.columns:
        X_new["Avg_Spend_Per_Purchase"] = X_new["Lifetime_Value"] / (X_new["Total_Purchases"] + 1)

    eng_cols = ["Login_Frequency", "Session_Duration_Avg", "Pages_Per_Session",
                "Email_Open_Rate", "Mobile_App_Usage", "Social_Media_Engagement_Score"]
    available = [c for c in eng_cols if c in X_new.columns]
    if len(available) > 0:
        X_new["Engagement_Index"] = X_new[available].mean(axis=1)

    if "Days_Since_Last_Purchase" in X_new.columns and "Membership_Years" in X_new.columns:
        denom = (X_new["Membership_Years"].clip(lower=0) * 365) + 1
        X_new["Recency_per_Year"] = X_new["Days_Since_Last_Purchase"] / denom

    X_new = X_new.replace([np.inf, -np.inf], np.nan)
    return X_new

feature_adder = FunctionTransformer(add_features, validate=False)

# -----------------------------
# Train model (cached)
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv(FILE_PATH)
    df[TARGET] = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Build FE preprocessor
    X_train_fe = add_features(X_train)
    num_cols_fe = X_train_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols_fe = X_train_fe.select_dtypes(include=["object"]).columns.tolist()

    preprocessor_fe = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols_fe),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols_fe)
    ])

    rf_fe_pipe = Pipeline([
        ("features", feature_adder),
        ("preprocess", preprocessor_fe),
        ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))
    ])

    # Optional tuning (kept light for Streamlit runtime)
    param_dist = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 10, 30]
    }

    search = RandomizedSearchCV(
        rf_fe_pipe,
        param_distributions=param_dist,
        n_iter=6,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # quick evaluation (for display only)
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)

    return best_model, X, test_f1

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 E-commerce Customer Churn Predictor")
st.write(
    "This app predicts whether a customer is likely to churn and provides a churn probability "
    "to support retention decisions."
)

with st.spinner("Training model (cached after first run)..."):
    model, X_all, test_f1 = train_model()

st.success(f"Model ready ✅ (Test F1 ≈ {test_f1:.3f})")

st.subheader("Customer Inputs")

# Build form from dataset columns (excluding target)
input_data = {}
for col in X_all.columns:
    if pd.api.types.is_numeric_dtype(X_all[col]):
        min_v = float(np.nanpercentile(X_all[col], 1))
        max_v = float(np.nanpercentile(X_all[col], 99))
        default_v = float(np.nanmedian(X_all[col]))
        input_data[col] = st.number_input(col, value=default_v, min_value=min_v, max_value=max_v)
    else:
        options = X_all[col].dropna().astype(str).unique().tolist()
        options = sorted(options)[:200]  # safety cap
        default = options[0] if len(options) > 0 else ""
        input_data[col] = st.selectbox(col, options=options, index=0)

input_df = pd.DataFrame([input_data])

st.divider()

if st.button("Predict churn risk"):
    pred = int(model.predict(input_df)[0])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_df)[0, 1])
    else:
        prob = None

    if pred == 1:
        st.error("Prediction: CHURN (High risk)")
    else:
        st.success("Prediction: NOT CHURN (Lower risk)")

    if prob is not None:
        st.write(f"Churn probability: **{prob:.2%}**")

    with st.expander("Show input data"):
        st.dataframe(input_df)
