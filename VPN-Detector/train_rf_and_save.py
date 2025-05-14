"""
Train a quick Random‑Forest on clean_data/vpn_15s.csv
and save to models/rf_model.pkl
Run: python train_rf_and_save.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# ⚙️ 1) Đọc dữ liệu – sửa tên file tương ứng dữ liệu của bạn
DATA_PATH = r"clean_data\vpn_15s_labeled.csv"        # đổi sang file bạn có
df = pd.read_csv(DATA_PATH)

# ⚙️ 2) Giả sử cột nhãn tên 'label'
#     0 = Non‑VPN, 1 = VPN
X = df.drop(columns=["label"])
y = df["label"]

# ⚙️ 3) Tách train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ⚙️ 4) Huấn luyện RF nhanh
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

print("\n=== Test report ===")
print(classification_report(y_test, rf.predict(X_test)))

# ⚙️ 5) Lưu mô hình
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_model.pkl")
print("\n✅ Saved model to models/rf_model.pkl")
