
import os
import sys
import time

print("Starting debug script...")
start_time = time.time()

try:
    print("Importing torch...")
    import torch
    print(f"Torch imported in {time.time() - start_time:.2f}s")
    
    print("Importing PechayPredictor...")
    from predict import PechayPredictor
    print(f"PechayPredictor imported in {time.time() - start_time:.2f}s")
    
    model_path = "pechay_cnn_model_20251212_184656.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        predictor = PechayPredictor(model_path, device='cpu')
        print(f"Model loaded successfully in {time.time() - start_time:.2f}s")
    else:
        print(f"Model file {model_path} not found!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Debug script finished.")
