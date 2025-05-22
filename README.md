## 🛡️ VPN-Detector Traffic Analysis

A machine learning-based system for detecting VPN traffic using network flow data and real-time Gradio app deployment via Docker.

---

## 📌 Problem Statement

Virtual Private Networks (VPNs) encrypt traffic to anonymize user identity or bypass restrictions.  
For network administrators, identifying VPN vs. non-VPN traffic is critical for enforcing security policies and detecting anomalies.

This project builds a robust detection model using statistical analysis and machine learning techniques.

---

## 📁 Dataset

We used the **ISCX VPN-nonVPN 2016** dataset. The original `.arff` files were cleaned and converted to `.csv` with multiple time window sizes:

```text
| File Name               | Rows    | Label       |
|------------------------|---------|-------------|
| vpn15s_cleaned.csv     | 19.5k   | VPN (1)     |
| novpn15s_cleaned.csv   | 17.9k   | Non-VPN (0) |
| vpn30s_cleaned.csv     | 15.4k   | VPN (1)     |
| novpn30s_cleaned.csv   | 13.8k   | Non-VPN (0) |
| vpn60s_cleaned.csv     | 13.8k   | VPN (1)     |
| novpn60s_cleaned.csv   | 17.1k   | Non-VPN (0) |
| vpn120s_cleaned.csv    | 11k     | VPN (1)     |
| novpn120s_cleaned.csv  | 10k     | Non-VPN (0) |

```

> Cleaned data is stored in the `clean_data/` directory.

---

## ⚙️ Methodology

### 🔄 Preprocessing
- Missing value imputation
- One-hot encoding (via `ColumnTransformer`)

### 🧠 Model Training
```text
| Time Window | Best Model |
|-------------|------------|
| 15s         | XGBoost    |
| 30s         | XGBoost    |
| 60s         | XGBoost    |
| 120s        | XGBoost    |
```

### 📊 Model Evaluation
- 5-fold cross-validation
- Hold-out test set: 30%
- Metrics: ROC-AUC, F1-score

### 📌 Feature Importance (Top 5)
- flowPktsPerSecond
- mean_fiat
- mean_biat
- duration
- pkt_len_mean

---

## 📦 Project Structure

```text
VPN-Detector/
├── clean_data/              # Cleaned datasets
├── gradio_app.py            # Gradio app for prediction
├── Dockerfile               # Docker image setup
├── train_rf_and_save.py     # Train & save model
├── scripts/                 # Other training scripts
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/vuvannam12082005/vpn-detector-traffic-analysis.git
cd vpn-detector-traffic-analysis/VPN-Detector

# Build the Docker image
docker build -t vpn-detector .

# Run the container
docker run -p 7860:7860 vpn-detector

# Open the app at:
http://localhost:7860
```

---

## 📈 Results (5-Fold Cross-Validation)

```text
| Time Window | Highest F1-Score | ROC-AUC |
|-------------|------------------|---------|
| 15 seconds  | 0.94             | 0.99    |
| 30 seconds  | 0.93             | 0.98    |
| 60 seconds  | 0.91             | 0.98    |
| 120 seconds | 0.92             | 0.98    |
```

---

## 🔁 Reproducibility

### Train XGBoost for 15s Window:
```bash
python scripts/train_rf_and_save.py
```

You can modify and re-run for other window sizes accordingly.

---

## 📚 References

- **Dataset:** [ISCX VPN-nonVPN 2016](https://www.unb.ca/cic/datasets/vpn.html)
- **Citation:** Kaggle VPN Classification
- **License:** MIT License – Free to use with attribution.
