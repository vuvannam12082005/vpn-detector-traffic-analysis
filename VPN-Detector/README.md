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
```

## 5. Quick Start
git clone https://github.com/vuvannam12082005/vpn-detector-traffic-analysis.git
cd vpn-detector-traffic-analysis
cd VPN-Detect
docker build -t vpn-detector .
docker run -p 7860:7860 vpn-detector    # open http://localhost:7860

## 6. Results (30 % unseen test)
Window	Best single F1	Ensemble F1	ROC-AUC
15 s	0.90 (RF)	0.95	0.98
30 s	0.91 (SVM)	0.95	0.98
60 s	0.93 (LGBM)	0.95	0.99
120 s	0.92 (Cat)	0.95	0.98

<div align="center"> <img src="shap_beeswarm.png" width="650"> </div>
Top-5 important features: fwd_pkt_per_sec, pkt_len_mean, tls_ja3_hash, …

## 7. How to Reproduce
# example: train 15 s Random Forest
python scripts/train_lr_rf_15s.py           # outputs models/rf_15s.pkl

# build ensemble (loads all *.pkl from models/)
python stacking_all.py                      # outputs models/ensemble.pkl

## 8. License
MIT – free to use with attribution. 
http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/
https://www.kaggle.com/code/vojtchschiller/vpn-non-vpn-traffic-classification
