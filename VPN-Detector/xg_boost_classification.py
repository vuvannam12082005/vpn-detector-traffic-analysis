import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Đọc dữ liệu
    yes_df = pd.read_csv('clean_data/vpn60s_cleaned.csv')
    no_df  = pd.read_csv('clean_data/novpn60s_cleaned.csv')

    # 2. Gán nhãn
    yes_df['label'] = 1
    no_df['label']  = 0

    # 3. Kết hợp dữ liệu
    combined_df = pd.concat([yes_df, no_df], ignore_index=True)

    # 4. Tách features và target
    X = combined_df.drop(columns=['label'])
    y = combined_df['label']

    # 5. Chia train/test (giữ tỷ lệ nhãn với stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 6. Xác định cột số và cột phân loại
    cate_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    nume_cols = [c for c in X_train.columns if X_train[c].dtype in ['int64', 'float64']]

    # 7. Tạo transformers cho preprocessing
    numerical_transformer = SimpleImputer(strategy='constant')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 8. Kết hợp preprocessing với ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, nume_cols),
        ('cat', categorical_transformer, cate_cols)
    ])

    # 9. Ép dtype về float32 (không bắt buộc với CPU, nhưng giữ để thống nhất)
    to_float32 = FunctionTransformer(lambda X: X.astype(np.float32))

    # 10. Định nghĩa XGBClassifier chạy trên CPU
    xgb_clf = XGBClassifier(
        tree_method='hist',           # histogram trên CPU
        predictor='cpu_predictor',    # predictor trên CPU
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # 11. Xây dựng pipeline hoàn chỉnh
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('cast', to_float32),
        ('model', xgb_clf)
    ])

    # 12. Huấn luyện model
    pipeline.fit(X_train, y_train)

    # 13. Dự đoán
    y_pred = pipeline.predict(X_test)

    # 14. Đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("First few predictions:", y_pred[:5])
    print("First few actual values:", y_test.values[:5])

    # 15. Vẽ ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No VPN', 'VPN'],
        yticklabels=['No VPN', 'VPN']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == '__main__':
    main()
