🛡️ VPN-Detector Traffic Analysis
Network traffic analysis using machine learning and statistics to detect VPN usage.

The project includes data cleaning, exploratory statistical analysis, model training/optimization (Random Forest, KNN, Decision Tree, Logistic Regression, XGBoost), 5-fold cross-validation, and a Dockerized Gradio app for real-time prediction.

1. Problem Discussion
Virtual Private Networks (VPNs) encrypt traffic to mask user identity or bypass restrictions. Network administrators need an automated method to identify VPN or non-VPN traffic to enforce policies or detect anomalies.

2. Dataset
We used the publicly available ISCX VPN-nonVPN 2016 dataset:

Downloaded in .arff format → cleaned and converted to .csv.
Four time window sizes: 15 seconds, 30 seconds, 60 seconds, 120 seconds.
Cleaned .csv files are stored in the clean_data/ directory.

File Name	Number of Rows	Label
vpn15s_cleaned.csv	19.5k	VPN (1)
novpn15s_cleaned.csv	17.9k	Non-VPN (0)
vpn30s_cleaned.csv	15.4k	VPN (1)
novpn30s_cleaned.csv	13.8k	Non-VPN (0)
vpn60s_cleaned.csv	13.8k	VPN (1)
novpn60s_cleaned.csv	17.1k	Non-VPN (0)
vpn120s_cleaned.csv	11k	VPN (1)
novpn120s_cleaned.csv	10k	Non-VPN (0)

3. Methodology

Preprocessing:
Imputation for missing values.
One-hot encoding for categorical variables (using ColumnTransformer).
Model Training for Each Window:
15 seconds → XGBoost
30 seconds → XGBoost
60 seconds → XGBoost
120 seconds → XGBoost
Model Evaluation:
Used 5-fold cross-validation.
Test data accounts for 30% (hold-out test set).
Evaluated using ROC-AUC and F1-score metrics.
Model Explainability:
Analyzed feature importance for the main models.

4. Project Structure

VPN-Detector/
├── clean_data/              # Cleaned CSV files
├── gradio_app.py            # Gradio app for user interface demo
├── Dockerfile               # Container build file
├── train_rf_and_save.py     # Script to train and save models
... (other model scripts and necessary files)
5. Quick Start Guide

git clone https://github.com/vuvannam12082005/vpn-detector-traffic-analysis.git
cd vpn-detector-traffic-analysis
cd VPN-Detect
docker build -t vpn-detector .
docker run -p 7860:7860 vpn-detector    # Open http://localhost:7860

6. Results (Using 5-fold CV)

Time Window	Highest F1-Score	ROC-AUC
15 seconds	0.94 (XGBoost)	0.99
30 seconds	0.93 (XGBoost)	0.98
60 seconds	0.91 (XGBoost/RF)	0.98
120 seconds	0.92 (XGBoost)	0.98
Top 5 Important Features: flowPktsPerSecond, mean_fiat, mean_biat, duration, pkt_len_mean.

7. How to Reproduce
# Example: Train XGBoost for the 15-second window
python scripts/train_rf_and_save.py

8. References
MIT – free to use with attribution.

Data Source: ISCX VPN-NonVPN 2016
Reference: Kaggle VPN Classification
