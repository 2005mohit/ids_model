import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import tempfile
import os
from collections import defaultdict

# Optional: PyShark for PCAP support
try:
    import pyshark
except ImportError:
    pyshark = None

# ================== Load Pipeline ==================
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS)")

pipeline = joblib.load("IDS_Pipeline_joblib.pkl")

model1 = pipeline['model1']
scaler1 = pipeline['scaler1']
imputer1 = pipeline['imputer1']

model2 = pipeline['model2']
scaler2 = pipeline['scaler2']
imputer2 = pipeline['imputer2']

le_attack = pipeline['le_attack']
feature_names = pipeline['feature_names']

# ================== PCAP to Feature Extraction ==================
def parse_pcap_to_features(pcap_file):
    if not pyshark:
        st.error("PyShark not installed. Run: pip install pyshark")
        return pd.DataFrame(columns=feature_names)

    cap = pyshark.FileCapture(pcap_file, only_summaries=False)
    flows = defaultdict(lambda: {
        'forward_packets': [],
        'timestamps': [],
    })

    for pkt in cap:
        try:
            src = pkt.ip.src
            dst = pkt.ip.dst
            sport = pkt[pkt.transport_layer].srcport if hasattr(pkt, 'transport_layer') else '0'
            dport = pkt[pkt.transport_layer].dstport if hasattr(pkt, 'transport_layer') else '0'
            protocol = pkt.transport_layer if hasattr(pkt, 'transport_layer') else 'UNKNOWN'

            flow_key = f"{src}-{dst}-{sport}-{dport}-{protocol}"
            size = int(pkt.length)
            time = float(pkt.sniff_timestamp)

            flows[flow_key]['timestamps'].append(time)
            flows[flow_key]['forward_packets'].append(size)
        except:
            continue

    rows = []
    for flow_key, data in flows.items():
        row = {name: 0 for name in feature_names}
        row['Destination Port'] = flow_key.split('-')[3]
        if data['timestamps']:
            row['Flow Duration'] = (max(data['timestamps']) - min(data['timestamps'])) * 1000
        if data['forward_packets']:
            row['Total Fwd Packets'] = len(data['forward_packets'])
            row['Total Length of Fwd Packets'] = sum(data['forward_packets'])
            row['Fwd Packet Length Max'] = max(data['forward_packets'])
            row['Fwd Packet Length Min'] = min(data['forward_packets'])
            row['Fwd Packet Length Mean'] = sum(data['forward_packets']) / len(data['forward_packets'])
        rows.append(row)

    df = pd.DataFrame(rows, columns=feature_names)
    df.fillna(0, inplace=True)
    return df

# ================== Prediction ==================
def predict_sample(sample_df):
    X1 = imputer1.transform(sample_df)
    X1 = scaler1.transform(X1)
    pred1 = model1.predict(X1)

    if pred1[0] == 0:
        return "✅ Normal Traffic"
    else:
        X2 = imputer2.transform(sample_df)
        X2 = scaler2.transform(X2)
        pred2 = model2.predict(X2)
        attack_label = le_attack.inverse_transform(pred2)[0]
        return f"🚨 Attack Detected: {attack_label}"

# ================== UI ==================
uploaded_file = st.file_uploader(
    "📂 Upload network traffic file (CSV/PCAP/PCAPNG)", 
    type=["csv", "pcap", "pcapng"]
)

if uploaded_file:
    temp_path = tempfile.mktemp(suffix=uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = os.path.splitext(uploaded_file.name)[-1].lower()

    if ext in [".pcap", ".pcapng"]:
        df = parse_pcap_to_features(temp_path)
        if df.empty:
            st.error("❌ Failed to extract data from PCAP/PCAPNG.")
            st.stop()
        st.info("PCAP/PCAPNG converted to feature dataframe.")
    else:
        df = pd.read_csv(temp_path)

    # Ensure all model features exist
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing features filled with 0: {missing_cols}")
        for col in missing_cols:
            df[col] = 0

    df = df[feature_names]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    st.success("✅ File processed successfully! Running IDS detection...")

    predictions = []
    for _, row in df.iterrows():
        sample = pd.DataFrame([row.values], columns=feature_names)
        predictions.append(predict_sample(sample))

    df["Prediction"] = predictions

    st.subheader("🔎 Predictions")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions", csv, "IDS_Predictions.csv", "text/csv")

