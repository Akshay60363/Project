import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import re

# ----------------------------------------------------------------------
# Text Preprocessing (same as during training)
# ----------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# ----------------------------------------------------------------------
# Load Trained Model and Artifacts
# ----------------------------------------------------------------------
@st.cache_resource
def load_trained_model(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with open(f"{model_dir}/mlb_classes.json", "r") as f:
        top_codes_list = json.load(f)
    mlb = MultiLabelBinarizer(classes=top_codes_list)
    mlb.fit([[]])
    return model, tokenizer, mlb

# ----------------------------------------------------------------------
# Predict ICD-9 Codes
# ----------------------------------------------------------------------
def predict_icd9(input_text: str, model, tokenizer, mlb, max_length=512, threshold=0.5):
    processed_text = preprocess_text(input_text)
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    y_pred = (probs > threshold).astype(int)
    predicted_codes = mlb.inverse_transform(np.array([y_pred]))
    return predicted_codes[0]

# ----------------------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------------------
st.title("ICD-9 Code Prediction")

model_dir = "D:\\ytopnoob\\project\\top10001\\final_mode4l" 

st.sidebar.header("Model Settings")
threshold = st.sidebar.slider("Prediction Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

st.write("Enter clinical text below to predict ICD-9 codes.")

input_text = st.text_area("Clinical Text", height=200)

if st.button("Predict"):
    if not input_text.strip():
        st.error("Please enter valid clinical text.")
    else:
        st.write("Loading model...")
        model, tokenizer, mlb = load_trained_model(model_dir)
        st.write("Predicting...")
        predicted_codes = predict_icd9(input_text, model, tokenizer, mlb, threshold=threshold)
        if predicted_codes:
            st.success("Predicted ICD-9 Codes:")
            st.write(predicted_codes)
        else:
            st.warning("No codes were predicted. Try lowering the threshold or using a different input.")
