import os
import glob
import numpy as np
import torch
from predict import PechayPredictor
import json

# Configuration
DATASET_PATH = r"c:\Users\Admin\trycnn\Dataset\Healthy Pechay.v1i.yolov9\train\images"
MODEL_PATH = "pechay_cnn_model_20251212_184656.pth"
OUTPUT_FILE = "healthy_embeddings.npy"

def generate_embeddings():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        predictor = PechayPredictor(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Scanning images in {DATASET_PATH}...")
    image_files = glob.glob(os.path.join(DATASET_PATH, "*.jpg")) + \
                  glob.glob(os.path.join(DATASET_PATH, "*.jpeg")) + \
                  glob.glob(os.path.join(DATASET_PATH, "*.png"))
    
    if not image_files:
        print("No images found!")
        return

    print(f"Found {len(image_files)} images.")
    
    embeddings = []
    valid_files = []
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}...", end="\r")
        features = predictor.extract_features(img_path)
        if features is not None:
            embeddings.append(features)
            valid_files.append(img_path)
            
    if embeddings:
        embeddings_array = np.array(embeddings)
        np.save(OUTPUT_FILE, embeddings_array)
        print(f"\nSaved {len(embeddings)} embeddings to {OUTPUT_FILE}")
        
        # Save filenames for reference if needed
        with open("healthy_files_index.json", "w") as f:
            json.dump(valid_files, f)
    else:
        print("\nFailed to extract any embeddings.")

if __name__ == "__main__":
    generate_embeddings()
