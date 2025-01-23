import logging
import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
from transformers import (
    LongformerTokenizer,
    LongformerForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import os
from PIL import Image
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.model.google import Gemini
# from phi.api import *
# import streamlit as st
from phi.tools.duckduckgo import DuckDuckGo
import json
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pdfplumber  
import google.generativeai as genai
import pandas as pd

# ---------------------- Load ICD-9 Descriptions -----------------------
icd9_desc_df = pd.read_csv("G:\\mimic 3\\D_ICD_DIAGNOSES.csv")
icd9_desc_df['ICD9_CODE'] = icd9_desc_df['ICD9_CODE'].astype(str)
icd9_descriptions = dict(zip(icd9_desc_df['ICD9_CODE'].str.replace('.', ''), icd9_desc_df['LONG_TITLE']))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from utils import (
    load_dataset,
    get_model_instance,
    load_checkpoint,
    can_load_checkpoint,
    normalize_text,
)

# ---------------------------------------------------------------------
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define device for PyTorch model
DEVICE = 'cpu'

# ----------------------- Google Gemini Setup --------------------------
google_api_key = "AIzaSyA24A6egT3L0NAKkkw9QHjfoizp7cJUTaA"
genai.configure(api_key='AIzaSyA24A6egT3L0NAKkkw9QHjfoizp7cJUTaA')

# Initialize the AI agent for image analysis
medical_agent = Agent(
    model=Gemini(api_key=google_api_key),
    tools=[DuckDuckGo()],
    markdown=True
)

# ---------------------------------------------------------------------
# ICD-9 code columns used during training (must match model output size)
icd9_columns = [
    '038.9', '244.9', '250.00', '272.0', '272.4', '276.1', '276.2', '285.1', '285.9',
    '287.5', '305.1', '311', '36.15', '37.22', '37.23', '38.91', '38.93', '39.61',
    '39.95', '401.9', '403.90', '410.71', '412', '414.01', '424.0', '427.31', '428.0',
    '486', '496', '507.0', '511.9', '518.81', '530.81', '584.9', '585.9', '599.0',
    '88.56', '88.72', '93.90', '96.04', '96.6', '96.71', '96.72', '99.04', '99.15',
    '995.92', 'V15.82', 'V45.81', 'V45.82', 'V58.61'
]

# ----------------------- Load ICD-9 Model (Version 1) ----------------
model_path_v1 = "G:\\backend\\clinical_longformer"
tokenizer_v1 = LongformerTokenizer.from_pretrained(model_path_v1)
model_v1 = LongformerForSequenceClassification.from_pretrained(model_path_v1)
model_v1.eval()  # Set model to evaluation mode

# ----------------------- Image Analysis Transforms --------------------
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size based on model's expected input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------
def load_model():
    """
    Loads your custom image-to-text model for generating captions or "reports".
    """
    print("Loading dataset and vocabulary...")
    dataset = load_dataset()  # Load dataset to access vocabulary
    vocabulary = dataset.vocab  # Assuming 'vocab' is an attribute of the dataset

    print("Initializing the model...")
    model = get_model_instance(vocabulary)  # Initialize the model

    if can_load_checkpoint():
        print("Loading checkpoint...")
        load_checkpoint(model)
    else:
        print("No checkpoint found, starting with untrained model.")

    model.eval()  # Set the model to evaluation mode
    print("Model is ready for inference.")
    return model

def preprocess_image(image_path):
    """
    Preprocess the input image for your custom model.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image = TRANSFORMS(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)

def generate_report(model, image_path):
    """
    Generates a textual 'report' or 'caption' for a given image using your custom model.
    """
    image = preprocess_image(image_path)
    with torch.no_grad():
        # Assuming the model has a 'generate_caption' method
        output = model.generate_caption(image, max_length=25)
        report = " ".join(output)
    return report

# ----------------------- Text Preprocessing ---------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# ----------------------- Loading a Trained Model (Version 2) ----------
def load_trained_model(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with open(f"{model_dir}/mlb_classes.json", "r") as f:
        top_codes_list = json.load(f)
    mlb = MultiLabelBinarizer(classes=top_codes_list)
    mlb.fit([[]])
    return model, tokenizer, mlb

# ----------------------- Prediction Functions -------------------------
def predict_icd9_v1(texts, tokenizer, model, threshold=0.5):
    """
    Multi-label ICD-9 prediction using the Longformer model (Version 1).
    """
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
    
    # Match with descriptions
    predictions_with_desc = []
    for codes in predicted_icd9:
        code_with_desc = [
            (
                code, 
                icd9_descriptions.get(code.replace('.', ''), "Description not found")
            ) 
            for code in codes
        ]
        predictions_with_desc.append(code_with_desc)
    
    return predictions_with_desc

def predict_icd9_v2(input_text: str, model, tokenizer, mlb, max_length=512, threshold=0.5):
    """
    Multi-label ICD-9 prediction using a second model (Version 2).
    """
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

# ----------------------- PDF Text Extraction --------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using pdfplumber.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ----------------------- API Endpoints -------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict ICD-9 codes (Version 1 model).
    """
    data = request.json
    texts = data.get("texts", [])
    threshold = float(data.get("threshold", 0.5))

    if not texts:
        return jsonify({"error": "No texts provided."}), 400

    predictions = predict_icd9_v1(texts, tokenizer_v1, model_v1, threshold)
    return jsonify({"predictions": predictions})



@app.route("/predict_icd9_new", methods=["POST"])
def predict_icd9_new():
    """
    Example endpoint using Google Gemini to generate ICD-9 codes from a note.
    """
    try:
        data = request.get_json()
        clinical_note = data.get('clinical_note', '')

        if not clinical_note:
            return jsonify({'error': 'No clinical note provided'}), 400
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        
        # Define the prompt
        prompt = f"""
        Based on the following clinical note, provide the relevant ICD-9 codes. 
        Include codes even if not confirmed. For normal findings, use code V72.5:
        {clinical_note}
        """

        # Use Gemini API
        response = model.generate_content(prompt)
        return jsonify({'predictions': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/icd9_predict', methods=['POST'])
def icd_predict():
    """
    Endpoint for your second model (Version 2).
    """
    data = request.get_json()
    input_text = data.get("clinical_text", "")
    threshold = data.get("threshold", 0.5)
    
    if not input_text.strip():
        return jsonify({"error": "Please enter valid clinical text."}), 400
    
    model_dir = "G:\\backend\\top10001\\final_mode4l"
    model, tokenizer, mlb = load_trained_model(model_dir)
    
    predicted_codes = predict_icd9_v2(input_text, model, tokenizer, mlb, threshold=threshold)
    
    if predicted_codes:
        return jsonify({"predicted_codes": predicted_codes})
    else:
        return jsonify({
            "message": "No codes were predicted. Try lowering the threshold or using a different input."
        }), 200

# -------------------- Corrected /analyze Endpoint ---------------------
@app.route("/analyze", methods=["POST"])
def analyze_image():
    """
    Endpoint to analyze an uploaded image (X-ray) using the AI agent,
    then predict ICD-9 codes from the generated text.
    """
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided."}), 400

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    try:
        # Convert the file to an image and run the medical agent
        with open(file_path, "rb") as f:
            image = Image.open(f)
            image_path = "temp_xray_image.png"
            image.save(image_path)

        # Use Gemini-based medical_agent to produce analysis text
        response = medical_agent.run(image_analysis_query, images=[image_path])
        analysis_text = response.content  # The text describing the X-ray findings

        # Predict ICD-9 codes from this analysis text (Version 1)
        threshold = 0.5
        icd9_predictions = predict_icd9_v1([analysis_text], tokenizer_v1, model_v1, threshold)

        # Cleanup
        os.remove(image_path)

        return jsonify({
            "analysis": analysis_text,
            "icd9_predictions": icd9_predictions
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --------------- Corrected /upload_image Endpoint ---------------------
@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    Endpoint to receive an image, generate a "report" via your custom model,
    and predict ICD-9 codes from that report.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided."}), 400

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, file.filename)
    file.save(image_path)

    try:
        # Generate a text "report" from the X-ray using your custom model
        model = load_model()
        report = generate_report(model, image_path)

        # Now feed that generated report text to predict ICD-9 codes (Version 1)
        threshold = 0.5
        icd9_predictions = predict_icd9_v1([report], tokenizer_v1, model_v1, threshold)

        return jsonify({
            "report": report,
            "icd9_predictions": icd9_predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """
    Endpoint to handle PDF uploads, extract text, and predict ICD-9 codes using model V1.
    """
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file."}), 400

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    pdf_path = os.path.join(upload_dir, file.filename)
    file.save(pdf_path)

    try:
        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_path)
        if not extracted_text:
            return jsonify({"error": "No text could be extracted from the PDF."}), 400

        # Predict ICD-9 codes using Version 1
        predictions_v1 = predict_icd9_v1([extracted_text], tokenizer_v1, model_v1, threshold=0.5)

        return jsonify({
            "extracted_text": extracted_text,
            "predictions_v1": predictions_v1
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# ------------------ Image Analysis Prompt -----------------------------
image_analysis_query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the X-ray image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings
"""

# ----------------------- Run the Flask App ---------------------------
if __name__ == "__main__":
    upload_directory = "uploads"
    if not os.path.exists(upload_directory):
        os.makedirs(upload_directory)
    app.run(debug=True)
