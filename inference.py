import os
import torch
import config
from utils import (
    load_dataset,
    get_model_instance,
    load_checkpoint,
    can_load_checkpoint,
    normalize_text,
)
from PIL import Image
import torchvision.transforms as transforms

# Define device
DEVICE = 'cpu'

# Define image transformations (adjust based on training setup)
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),  # Replace with your model's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    """
    Loads the model with the vocabulary and checkpoint.
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
    Preprocess the input image for the model.
    """
    print(f"Preprocessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image = TRANSFORMS(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)


def generate_report(model, image_path):
    """
    Generates a report for a given image using the model.
    """
    image = preprocess_image(image_path)

    print("Generating report...")
    with torch.no_grad():
        # Assuming the model has a 'generate_caption' method
        output = model.generate_caption(image, max_length=25)
        report = " ".join(output)

    print(f"Generated report: {report}")
    return report


if __name__ == "__main__":
    # Path to the checkpoint file
    CHECKPOINT_PATH = config.CHECKPOINT_FILE  # Ensure config.CHECKPOINT_FILE is correctly set

    # Path to the input image
    IMAGE_PATH = "D:\\NLMCXR_png\\CXR3_IM-1384-1001.png"  # Replace with your image path

    # Load the model
    model = load_model()

    # Ensure the image exists before inference
    if os.path.exists(IMAGE_PATH):
        report = generate_report(model, IMAGE_PATH)
        print("Final Report:", report)
    else:
        print(f"Image not found at path: {IMAGE_PATH}")
