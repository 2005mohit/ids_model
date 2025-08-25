import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pathlib
import tempfile
import os
import subprocess
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# ================== Optional: PyShark ==================
try:
    import pyshark
except ImportError:
    pyshark = None

# ================== Fix for Streamlit Cloud ==================
TSHARK_PATH = "/usr/bin/tshark"
if pyshark and os.path.exists(TSHARK_PATH):
    os.environ["TSHARK_PATH"] = TSHARK_PATH
else:
    pyshark = None

# ================== Streamlit Config ==================
st.set_page_config(page_title="Intrusion Detection System", layout="wide")
st.title("🛡️ Intrusion Detection System (IDS)")

# ================== Load Pipeline ==================
MODEL_PATH = pathlib.Path(__file__).parent / "IDS_Pipeline_joblib.pkl"
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Could not load model pipeline: {e}")
    st.stop()

model1 = pipeline['model1']
scaler1 = pipeline['scaler1']
imputer1 = pipeline['imputer1']

model2 = pipeline['model2']
scaler2 = pipeline['scaler2']
imputer2 = pipeline['imputer2']

le_attack = pipeline['le_attack']
feature_names = pipeline['feature_names']

# ================== PCAP Parsing with PyShark ==================
def parse_pcap_with_pyshark(pcap_file):
    try:
        cap = pyshark.FileCapture(
            pcap_file,
            only_summaries=False,
            tshark_path=TSHARK_PATH
        )
        flows = defaultdict(lambda: {'forward_packets': [], 'timestamps': []})

        for pkt in cap:
            try:
                src = pkt.ip.src
                dst = pkt.ip.dst
                sport = getattr(pkt[pkt.transport_layer], "srcport", "0") if pkt.transport_layer else "0"
                dport = getattr(pkt[pkt.transport_layer], "dstport", "0") if pkt.transport_layer else "0"
                protocol = pkt.transport_layer if pkt.transport_layer else "UNKNOWN"

                flow_key = f"{src}-{dst}-{sport}-{dport}-{protocol}"
                size = int(pkt.length)
                time = float(pkt.sniff_timestamp)

                flows[flow_key]['timestamps'].append(time)
                flows[flow_key]['forward_packets'].append(size)
            except Exception:
                continue

        cap.close()

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

    except Exception as e:
        st.warning(f"⚠️ PyShark parsing failed: {e}")
        return pd.DataFrame(columns=feature_names)

# ================== PCAP Parsing with Tshark/Dumpcap ==================
def parse_pcap_with_tshark(pcap_file):
    try:
        csv_out = pcap_file + ".csv"
        cmd = [
            "tshark", "-r", pcap_file,
            "-T", "fields",
            "-e", "frame.time_epoch",
            "-e", "ip.src",
            "-e", "ip.dst",
            "-e", "frame.len",
            "-E", "header=y", "-E", "separator=,"
        ]
        with open(csv_out, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=True)

        df = pd.read_csv(csv_out)
        # NOTE: Map tshark fields → IDS features
        # Minimal mapping example:
        df_features = pd.DataFrame(columns=feature_names)
        df_features["Destination Port"] = 0
        df_features["Flow Duration"] = 0
        df_features["Total Fwd Packets"] = 1
        df_features["Total Length of Fwd Packets"] = df["frame.len"]
        df_features["Fwd Packet Length Max"] = df["frame.len"]
        df_features["Fwd Packet Length Min"] = df["frame.len"]
        df_features["Fwd Packet Length Mean"] = df["frame.len"]
        df_features.fillna(0, inplace=True)
        return df_features

    except Exception as e:
        st.warning(f"⚠️ Tshark/Dumpcap parsing failed: {e}")
        return pd.DataFrame(columns=feature_names)

# ================== Hybrid PCAP Parser ==================
def parse_pcap_to_features(pcap_file):
    if pyshark:
        df = parse_pcap_with_pyshark(pcap_file)
        if not df.empty:
            return df
    # Fallback to Tshark
    df = parse_pcap_with_tshark(pcap_file)
    if not df.empty:
        return df
    st.error("❌ Could not parse PCAP/PCAPNG. Please upload CSV instead.")
    return pd.DataFrame(columns=feature_names)

# ================== Prediction ==================
def predict_samples(df):
    try:
        X1 = imputer1.transform(df)
        X1 = scaler1.transform(X1)
        pred1 = model1.predict(X1)

        results = []
        for i, p in enumerate(pred1):
            if p == 0:
                results.append("✅ Normal Traffic")
            else:
                X2 = imputer2.transform([df.iloc[i]])
                X2 = scaler2.transform(X2)
                pred2 = model2.predict(X2)
                attack_label = le_attack.inverse_transform(pred2)[0]
                results.append(f"🚨 Attack Detected: {attack_label}")
        return results
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return ["Error"] * len(df)

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
            st.stop()
        st.info("✅ PCAP/PCAPNG converted to feature dataframe.")
    else:
        try:
            df = pd.read_csv(temp_path)
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            st.stop()

    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing features filled with 0: {missing_cols}")
        for col in missing_cols:
            df[col] = 0

    df = df[feature_names]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    st.success("✅ File processed successfully! Running IDS detection...")

    predictions = predict_samples(df)
    df["Prediction"] = predictions

    st.subheader("🔎 Predictions")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions", csv, "IDS_Predictions.csv", "text/csv")
