import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import os

# ⚙️ 1) Đọc dữ liệu 
DATA_PATH = "clean_data/vpn15s_labeled.csv"  # Sử dụng forward slash

df = pd.read_csv(DATA_PATH)

# ⚙️ 2) Giả sử cột nhãn tên 'label'
#     0 = Non‑VPN, 1 = VPN
X = df.drop(columns=["label"])
y = df["label"]

# ⚙️ 3) Tách train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ⚙️ 4) Huấn luyện XGBoost nhanh
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    use_label_encoder=False,  # Tránh cảnh báo
    eval_metric='logloss'    # Tránh cảnh báo
)
xgb.fit(X_train, y_train)

print("\n=== Test report ===")
print(classification_report(y_test, xgb.predict(X_test)))

# ⚙️ 5) Lưu mô hình
os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "models/xgb_model.pkl")
print("\n Đã lưu mô hình vào models/xgb_model.pkl")