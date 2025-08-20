import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# ================== Load Pipeline ==================
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS)")

# Load dict pipeline
pipeline = joblib.load("IDS_Pipeline_joblib.pkl")

# Extract objects
model1 = pipeline['model1']       # Binary classifier
scaler1 = pipeline['scaler1']
imputer1 = pipeline['imputer1']

model2 = pipeline['model2']       # Attack type classifier
scaler2 = pipeline['scaler2']
imputer2 = pipeline['imputer2']

le_attack = pipeline['le_attack'] # Label encoder
feature_names = pipeline['feature_names']


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
  
    # Strip whitespace and standardize column names in user file
    df.columns = df.columns.str.strip().str.lower()

    # Similarly normalize expected feature names
    feature_names_normalized = [col.strip().lower() for col in feature_names]

    # Check for missing features
    missing_cols = [orig for orig, norm in zip(feature_names, feature_names_normalized) if norm not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required features in uploaded CSV: {missing_cols}")
    else:
         selected_columns = []
         for f_native, f_norm in zip(feature_names, feature_names_normalized):
             # Find matching df column for each feature
             match = [col for col in df.columns if col == f_norm]
             if match:
                selected_columns.append(match[0])
            
        # Keep only required features
         df = df[selected_columns]

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
