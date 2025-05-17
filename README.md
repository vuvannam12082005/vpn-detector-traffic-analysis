# 🛡️ VPN-Detector Traffic Analysis

Machine-learning & statistical analysis of network flows to **detect VPN usage**.  
Project covers data cleaning, exploratory stats, model training/tuning (RF, SVM,
XGBoost, 1-D CNN), ensemble stacking, SHAP explainability, and a **Dockerised Gradio
app** for real-time inference.

---

## 1. Problem Discussion
Virtual Private Networks (VPN) tunnel traffic to hide user identity or bypass
restrictions. Network administrators need an automatic way to recognise whether
a flow is VPN or non-VPN in order to enforce policy or spot anomalies.

## 2. Dataset
We use the public **ISCX VPN-nonVPN 2016** dataset:

* `.pcap` → converted to flow features via **CICFlowMeter**  
* Four window sizes: **15 s, 30 s, 60 s, 120 s**  
* Cleaned CSVs stored in `clean_data/`

| File | #Rows | Label |
|------|-------|-------|
| `vpn15s_cleaned.csv` | 23 k | VPN (1) |
| `novpn15s_cleaned.csv` | 22 k | Non-VPN (0) |
| … | … | … |

## 3. Methodology
1. **Pre-processing** – impute, one-hot categoricals (`ColumnTransformer`).
2. Train individual models per window  
   * 15 s → Logistic Regression & Random Forest  
   * 30 s → SVM (RBF) & K-NN  
   * 60 s → XGBoost & LightGBM (Optuna tuned)  
   * 120 s → CatBoost & 1-D CNN
3. **Ensemble** – Stacking (Logistic meta-learner) over 8 base models.
4. **Model assessment** – 5-fold CV, hold-out 30 % test, ROC-AUC / F1 ± 95 % CI.
5. **Explainability** – SHAP beeswarm & force plot on final ensemble.

## 4. Project Structure
```text
VPN-Detector/
├── clean_data/              # cleaned CSVs
├── models/                  # *.pkl / .h5 saved models
├── scripts/                 # training scripts per window
├── stacking_all.py          # build final ensemble
├── gradio_app.py            # UI demo
├── explain.ipynb            # SHAP analysis
├── Dockerfile               # container build
└── README.md


.....(5-9)
