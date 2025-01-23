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
    """
    Perform advanced text cleaning:
      - Convert to lowercase
      - Remove bracketed deidentifications [**...**]
      - Remove excessive punctuation
      - Convert multiple spaces/newlines to single space
      - Strip whitespace
    """
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)  # remove deidentified brackets
    text = re.sub(r"([!?.,])\1+", r"\1", text)    # collapse repeated punctuation
    text = re.sub(r"[\r\n\t]+", " ", text)        # collapse lines/tabs to space
    text = re.sub(r"\s+", " ", text)             # multiple spaces -> single
    text = text.strip()
    return text

# ----------------------------------------------------------------------
# Load Trained Model and Artifacts
# ----------------------------------------------------------------------
def load_trained_model(model_dir: str):
    """
    Load the trained model, tokenizer, and MultiLabelBinarizer.
    """
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # Set to evaluation mode

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the MultiLabelBinarizer
    with open(f"{model_dir}/mlb_classes.json", "r") as f:
        top_codes_list = json.load(f)
    mlb = MultiLabelBinarizer(classes=top_codes_list)
    mlb.fit([[]])  # Initialize the binarizer

    return model, tokenizer, mlb

# ----------------------------------------------------------------------
# Predict ICD-9 Codes
# ----------------------------------------------------------------------
def predict_icd9(input_text: str, model, tokenizer, mlb, max_length=512, threshold=0.5):
    """
    Predict ICD-9 codes for a given clinical text.
    """
    # Preprocess the input text
    processed_text = preprocess_text(input_text)

    # Tokenize the input
    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    # Perform inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Apply threshold to get predicted labels
    y_pred = (probs > threshold).astype(int)

    # Decode the predicted labels back to ICD-9 codes
    predicted_codes = mlb.inverse_transform(np.array([y_pred]))  # Ensure 2D array

    return predicted_codes[0]  # Return as a list of codes

# ----------------------------------------------------------------------
# Inference Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Directory where the model and artifacts are saved
    model_dir = "./"

    # Load the model and related artifacts
    model, tokenizer, mlb = load_trained_model(model_dir)

    # Example input text
    input_text = """Acute nasopharyngitis , true acute abnormality    """

    # Predict ICD-9 codes
    predicted_codes = predict_icd9(input_text, model, tokenizer, mlb, threshold= 0.2)

    # Print the predicted ICD-9 codes
    print("Predicted ICD-9 Codes:", predicted_codes)
