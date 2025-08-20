import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# =====================
# Load Pipeline + Feature Names
# =====================
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("IDS_Pipeline_joblib.pkl")
        feature_names = joblib.load("features_names.pkl")
        return pipeline, feature_names
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        st.stop()

pipeline, feature_names = load_pipeline()

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Cyber IDS Dashboard", layout="wide")

# Cyber theme
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Courier New', monospace;
}
h1, h2, h3, h4 {
    color: #00ffe5;
}
.stButton>button {
    background-color: #00ffe5;
    color: #000;
    font-weight: bold;
    border-radius: 10px;
}
.stDownloadButton>button {
    background-color: #ff00ff;
    color: #fff;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align:center; padding:20px'>
    <h1>🔐 Intrusion Detection System (IDS)</h1>
    <p style='color:#a0f7f7;'>Detect BENIGN or ATTACK network traffic automatically from CSV input</p>
</div>
""", unsafe_allow_html=True)

# =====================
# Upload Section
# =====================
with st.container():
    st.markdown("<h3 style='color:#00ffe5'>📁 Upload Your CSV File</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose CSV file with network traffic features", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Validate columns
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                st.error(f"CSV is missing required features: {missing_features}")
            else:
                predictions = []
                for i in range(df.shape[0]):
                    features = df[feature_names].iloc[i].values
                    pred = pipeline.predict(features)
                    predictions.append(pred)
                df["Prediction"] = predictions

                # Highlight predictions
                def highlight_prediction(row):
                    color = '#00ff7f' if row['Prediction']=='BENIGN' else '#ff4d4d'
                    return ['background-color: {}'.format(color) if col=='Prediction' else '' for col in row.index]

                st.markdown("<h3 style='color:#00ffe5'>🛡️ Predictions (Preview)</h3>", unsafe_allow_html=True)
                st.dataframe(df[["Prediction"]].head(20).style.apply(highlight_prediction, axis=1), height=400)

                # Download full results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Predictions CSV",
                    data=csv,
                    file_name="IDS_Predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
