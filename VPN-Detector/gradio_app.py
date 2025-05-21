import gradio as gr
import pandas as pd
import joblib
import os

# Load mô hình
MODEL_PATH = os.path.join("models", "xgb_model.pkl")  # Đổi từ rf_model.pkl thành xgb_model.pkl
try:
    model = joblib.load(MODEL_PATH)
    print(f"Đã tải mô hình từ {MODEL_PATH}")
except FileNotFoundError as e:
    print(f"Lỗi: {e}. Vui lòng chạy train_xgb_and_save.py để tạo mô hình.")
    exit(1)

# Hàm dự đoán
def predict(file):
    if file is None:
        return "No file", None
    df = pd.read_csv(file.name)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    preds = model.predict(df)
    df["prediction"] = preds
    summary = {
        "Total": len(preds),
        "VPN (1)": int((preds == 1).sum()),
        "Non-VPN (0)": int((preds == 0).sum()),
    }
    return summary, df

# Tạo giao diện Gradio
with gr.Blocks(title="VPN Detector") as demo:
    gr.Markdown("# 🛡️ VPN Detector\nUpload CSV flows -> model predicts VPN or not.")
    inp = gr.File(label="Upload CSV")
    out_summary = gr.JSON(label="Summary")
    out_table = gr.Dataframe(label="Detail")
    btn = gr.Button("Predict")
    btn.click(predict, inp, [out_summary, out_table])

# Khởi động server với thông báo tùy chỉnh
if __name__ == "__main__":
    print("\n⚠️ Truy cập từ máy bạn tại: http://127.0.0.1:7860", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)