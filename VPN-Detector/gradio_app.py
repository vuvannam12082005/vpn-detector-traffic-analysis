"""
Simple Gradio UI – upload CSV -> get VPN prediction (0/1)
Run:  python gradio_app.py
"""

import gradio as gr
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "rf_model.pkl")
model = joblib.load(MODEL_PATH)

def predict(file):
    if file is None:
        return "No file"
    df = pd.read_csv(file.name)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    preds = model.predict(df)
    df["prediction"] = preds
    summary = {
        "Total": len(preds),
        "VPN (1)": int((preds == 1).sum()),
        "Non‑VPN (0)": int((preds == 0).sum()),
    }
    return summary, df

with gr.Blocks(title="VPN Detector") as demo:
    gr.Markdown("# 🛡️ VPN Detector\nUpload CSV flows -> model predicts VPN or not.")
    inp = gr.File(label="Upload CSV")
    out_summary = gr.JSON(label="Summary")
    out_table = gr.Dataframe(label="Detail")
    btn = gr.Button("Predict")
    btn.click(predict, inp, [out_summary, out_table])

if __name__ == "__main__":
    demo.launch()
