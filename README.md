# ML-Based Intrusion Detection System

## 1. Dataset

The project utilizes the **CICIDS 2017** dataset, a benchmark dataset widely adopted for network intrusion detection research. This dataset comprises labeled network traffic flows capturing various attack scenarios and benign network activity.

- **Source:** [CICIDS 2017 Dataset](http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip) (224 MB zipped)
- **Structure:** Multiple CSV files representing traffic flows for different days and scenarios were combined into a single master dataset.
- **Size:** 2,830,743 rows × 79 features (including the label column) before cleaning.
- **Features:** Include network flow statistics, packet lengths, timing intervals, and protocol flags.
- **Labels:** The dataset covers the following classes:
  - `BENIGN` (normal traffic)
  -  Attack types:
    - DoS Hulk
    - DoS GoldenEye
    - DoS Slowloris
    - DoS Slowhttptest
    - DDoS
    - PortScan
    - Bot
    - FTP-Patator
    - SSH-Patator
    - Web Attacks (Brute Force, SQL Injection, XSS)
    - Infiltration
    - Heartbleed

***

## 2. Exploratory Data Analysis (EDA)

### Overview

- The combined dataset had 2,830,743 flows and 79 columns with 54 integer, 24 float, and 1 categorical (label) feature.
- Minimal missing data—only `Flow Bytes/s` has missing values at 0.05%.

### Class Distribution

- Strong class imbalance with benign traffic representing ~80.3% of samples; attacks account for ~19.7%.
- DoS Hulk, PortScan, and DDoS attacks dominate the attack instances, while many other attack types are rare.

### Statistical Summary & Outliers

- Features exhibit skewed, heavy-tailed distributions with wide ranges and extreme outliers, especially in timing and packet/byte count features.
- Some feature anomalies such as negative header length values may require domain knowledge for handling or removal.
- Outlier presence is expected given the nature of attack patterns but requires careful treatment.

### Correlation Clustermap

- Hierarchical clustering identifies distinct groups of strongly correlated features such as packet length statistics and inter-arrival timing metrics.
- This informs dimensionality reduction and feature selection approaches in later modeling.

### Feature Distributions by Label

- Boxplots of key features (e.g., Flow Duration and Forward Packets/s) across class labels highlight separability trends with attack types often showing larger, more variable values.
- However, significant overlap remains, especially in scarce classes, indicating classification complexity.

***

## 3. Data Preprocessing

- Multiple raw CSVs were combined into a `combined_data.csv`.
- The label column was extracted and saved separately before encoding.
- Label encoding was applied with `BENIGN` as 0 and attack classes numerically encoded.
- Missing values in `Flow Bytes/s` were handled by dropping incomplete rows.
- Constant-valued columns, duplicate columns, and duplicate rows were removed.
- After cleaning, the dataset shape reduced to **2,522,009 rows × 66 features**, free of missing values and ready for modeling.

***

## 4. Model Architecture and Performance

The proposed system uses a **two-stage pipeline** combining:

- **Binary Classification Model:** Distinguishes benign traffic from attacks.
- **Multi-Class Classification Model:** Identifies specific attack types for flows detected as attacks.

**Binary classifier:** An **XGBoost classifier** configured with 500 trees, max-depth 6, learning rate 0.05, and subsampling strategies, with built-in imbalance handling. It achieves near-perfect accuracy and an ROC-AUC score of 0.999978 on the test set.

**Multi-class classifier:** A **LightGBM classifier** with 800 estimators, balanced class weights, and leaf-wise growth predicts attack types. It achieves high precision, recall, and F1 scores across most classes, with slight drops in rare categories.

***

## 5. Model Training

### Binary Classification Training

- Attack classes were grouped as `1`, benign as `0`.
- Data split 80-20 with stratification.
- Infinite values in features replaced with NaN before imputing with mean of training data.
- Data standardized with `StandardScaler`.
- Final model saved with corresponding scaler and imputer using `joblib`.

### Multi-Class Classification Training

- Benign samples filtered out.
- Attack labels mapped to names, then re-encoded.
- Data split 80-20 with stratification on attack classes.
- Infinite values handled similarly by imputation.
- Features standardized separately with a new scaler.
- Model trained using LightGBM with balanced classes and saved with all preprocessing components.

***

## 6. Model Deployment

### Pipeline Wrapper

The two-stage intrusion detection system is wrapped in a single `IDSPipeline` class, integrating preprocessing and models for streamlined predictions.

- **Components:**  
  - Preprocessing: two `SimpleImputer` and two `StandardScaler` instances (one each for binary and multi-class models).  
  - Models: `XGBClassifier` (binary) and `LGBMClassifier` (multi-class).  
  - Label encoder to decode multi-class attack labels.  
  - Feature names to ensure consistent input.

- **Workflow:**  
  1. Input features are preprocessed (imputed + scaled) for binary classification to predict benign vs. attack.  
  2. Flows predicted as attacks are further preprocessed and passed to multi-class classifier to identify attack type.  
  3. Results from both models are combined and returned in a user-friendly format.

- **Deployment:**  
  The entire pipeline is saved as `IDS_Pipeline.pkl` for easy loading and use in the Streamlit app, enabling real-time network traffic detection and classification through a consistent, end-to-end interface.


### Web Application

- Deployed on Streamlit Cloud platform, linked to GitHub repository.
- The app accepts user inputs as CSV, PCAP, or PCAPNG files.
- Features are extracted in real-time and passed through the pipeline.
- Binary model detects benign vs. attack flows.
- Attack flows routed to multi-class model for detailed classification.
- Results previewed on app and downloadable as CSV for user convenience.

- The model was successfully tested on real WiFi traffic, showing high precision in detection.

***

## 7. Limitations and Conclusion

- Scarce attack classes (e.g., Heartbleed) remain challenging to detect accurately due to insufficient data.
- Future work aims to improve detection for rare classes via data augmentation, enhanced feature engineering, and more robust model architectures.
- UI improvements and real-time deployment optimizations are planned to facilitate network security monitoring.
- Overall, the project demonstrates a scalable and effective ML-based approach to multi-class intrusion detection on large network flow datasets.

***

*This completes the IDS model development journey—leveraging high-quality data, advanced preprocessing, two-stage modeling, and cloud deployment to help secure real-world networks.*
