import streamlit as st
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "./clinical_longformer"
tokenizer = LongformerTokenizer.from_pretrained(model_path)
model = LongformerForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

# ICD-9 code columns used during training
icd9_columns = [
    '038.9', '244.9', '250.00', '272.0', '272.4', '276.1', '276.2', '285.1', '285.9',
    '287.5', '305.1', '311', '36.15', '37.22', '37.23', '38.91', '38.93', '39.61',
    '39.95', '401.9', '403.90', '410.71', '412', '414.01', '424.0', '427.31', '428.0',
    '486', '496', '507.0', '511.9', '518.81', '530.81', '584.9', '585.9', '599.0',
    '88.56', '88.72', '93.90', '96.04', '96.6', '96.71', '96.72', '99.04', '99.15',
    '995.92', 'V15.82', 'V45.81', 'V45.82', 'V58.61'
]

# Function for making predictions
def predict_icd9(texts, tokenizer, model, threshold=0.5):
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).int()
    
    predicted_icd9 = []
    for pred in predictions:
        codes = [icd9_columns[i] for i, val in enumerate(pred) if val == 1]
        predicted_icd9.append(codes)
    
    return predicted_icd9

# Streamlit UI
st.title("ICD-9 Code Prediction")
st.sidebar.header("Model Options")
model_option = st.sidebar.selectbox("Select Model", [ "ClinicalLongformer"])
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

st.write("### Enter Medical Summary")
input_text = st.text_area("Medical Summary", placeholder="Enter clinical notes here...")

if st.button("Predict"):
    if input_text.strip():
        predictions = predict_icd9([input_text], tokenizer, model, threshold)
        st.write("### Predicted ICD-9 Codes")
        for code in predictions[0]:
            st.write(f"- {code}")
    else:
        st.error("Please enter a medical summary.")
