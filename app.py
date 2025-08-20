import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# ================== Page Config ==================
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS)")

# ================== Load Pipeline ==================
pipeline = joblib.load("IDS_Pipeline_joblib.pkl")

# Extract pipeline components
model1 = pipeline['model1']       # Binary classifier
scaler1 = pipeline['scaler1']
imputer1 = pipeline['imputer1']

model2 = pipeline['model2']       # Attack type classifier
scaler2 = pipeline['scaler2']
imputer2 = pipeline['imputer2']

le_attack = pipeline['le_attack'] # Label encoder
feature_names = pipeline['feature_names']


# ================== Helper Function ==================
def ensure_features(df, feature_names):
    """
    Ensure uploaded CSV has all required features for the model.
    Missing features -> auto-filled with 0.
    Extra features -> dropped.
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]  # Reorder and drop extras
    return df


# ================== Prediction Function ==================
def predict_sample(sample_df):
    """Predict whether traffic is normal or an attack"""
    # Binary classification
    X1 = imputer1.transform(sample_df)
    X1 = scaler1.transform(X1)
    pred1 = model1.predict(X1)

    if pred1[0] == 0:
        return "✅ Normal Traffic"
    else:
        # Attack type classification
        X2 = imputer2.transform(sample_df)
        X2 = scaler2.transform(X2)
        pred2 = model2.predict(X2)
        attack_label = le_attack.inverse_transform(pred2)[0]
        return f"🚨 Attack Detected: {attack_label}"


# ================== Streamlit UI ==================
uploaded_file = st.file_uploader("📂 Upload CSV file with network traffic data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Fix columns to match training features
    df = ensure_features(df, feature_names)

    st.success("✅ File uploaded successfully! Processing data...")

    # Predictions
    results = []
    for _, row in df.iterrows():
        sample = pd.DataFrame([row.values], columns=feature_names)
        result = predict_sample(sample)
        results.append(result)

    df["Prediction"] = results

    # Show results
    st.subheader("🔎 Predictions")
    st.dataframe(df)

    # Option to download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions", csv, "IDS_Predictions.csv", "text/csv")
