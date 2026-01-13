from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash, make_response, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import random
import json
import uuid
from datetime import datetime
import threading
import requests
import base64
from PIL import Image
import numpy as np
try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, sobel
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Digital image detection will be limited.")

# Import DB functions early to avoid race conditions in background threads
from db import (
    get_user_by_username, get_user_by_email, create_user, verify_user_credentials,
    create_detection_result, get_detection_result_by_filename,
    get_all_detection_results, get_detection_results_by_condition,
    delete_detection_result, update_detection_result, get_dashboard_stats, DatabaseError,
    update_user_email, update_user_password, delete_user,
    create_dataset_image, get_dataset_images, get_dataset_stats,
    create_yolo_file, create_file_upload_log, upload_image_to_storage,
    save_custom_training_data, get_custom_training_data,
    save_dataset_entry, get_dataset_entries,
    get_yolo_file_by_filename, get_yolo_files_by_condition, get_all_yolo_files
)
from training import train_yolo_model

# Import face recognition style detection
try:
    from face_recognition_style_detection import face_recognition_style_match
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[INFO] Face recognition style detection module not available")

# Lazy load PechayPredictor to avoid blocking startup if torch is slow/broken
cnn_predictor = None
PechayPredictor = None
yolo_model = None

HEALTHY_ONLY_MODE = os.getenv("HEALTHY_ONLY_MODE", "1").strip().lower() in ("1", "true", "yes", "on")
YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "").strip()
CNN_MODEL_PATHS = [p.strip() for p in os.getenv("CNN_MODEL_PATHS", "").split(";") if p.strip()]
YOLO_CONF_MATCHED = float(os.getenv("YOLO_CONF_MATCHED", "0.25") or 0.25)
YOLO_CONF_UNKNOWN = float(os.getenv("YOLO_CONF_UNKNOWN", "0.35") or 0.35)
STORAGE_BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", "petchay-images").strip() or "petchay-images"

ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "bael")
ROBOFLOW_WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW", "classify-and-conditionally-detect")


def load_model_background():
    global cnn_predictor, PechayPredictor, yolo_model
    print("Background thread: Attempting to load model...")
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        yolo_weight_candidates = []
        if YOLO_WEIGHTS_PATH:
            yolo_weight_candidates.append(YOLO_WEIGHTS_PATH)
        yolo_weight_candidates.extend([
            os.path.join(base_dir, "petchay_detection", "petchay_model", "weights", "best.pt"),
            os.path.join(base_dir, "weights", "best.pt"),
            os.path.join(base_dir, "best.pt"),
        ])
        for weight_path in yolo_weight_candidates:
            try:
                if os.path.exists(weight_path):
                    from ultralytics import YOLO
                    yolo_model = YOLO(weight_path)
                    print(f"YOLO model loaded successfully from {weight_path}!")
                    break
            except Exception as e:
                print(f"Error loading YOLO model from {weight_path}: {e}")
                continue

        from predict import PechayPredictor as PP
        PechayPredictor = PP
        
        model_candidates = CNN_MODEL_PATHS or [
            "pechay_cnn_model_20251212_184656.pth",
            "pechay_cnn_model_20251114_155348.pth",
            "pechay_cnn_model_20251023_111926.pth",
            "pechay_cnn_model.pth"
        ]
        
        for model_path in model_candidates:
            try:
                if os.path.exists(model_path):
                    # Check file size - valid PyTorch models should be at least 1MB
                    file_size = os.path.getsize(model_path)
                    if file_size < 1024 * 1024:  # Less than 1MB
                        print(f"Skipping {model_path}: file too small ({file_size} bytes), likely corrupted")
                        continue
                    
                    cnn_predictor = PechayPredictor(model_path, device='cpu')
                    print(f"CNN model loaded successfully from {model_path}!")
                    break
            except Exception as e:
                print(f"Error loading CNN model from {model_path}: {e}")
                continue
                
        if cnn_predictor is None:
             print("No valid trained CNN model found in background. Using simulation.")
             
    except ImportError:
        print("Warning: Could not import PechayPredictor in background. Using simulation.")
    except Exception as e:
        print(f"Error in background model loader: {e}")

# Global variable for healthy embeddings
HEALTHY_EMBEDDINGS = None
# Global variables for custom reference data
REFERENCE_EMBEDDINGS = None
REFERENCE_LABELS = []

def load_embeddings():
    global HEALTHY_EMBEDDINGS, REFERENCE_EMBEDDINGS, REFERENCE_LABELS
    try:
        # Load standard healthy embeddings
        emb_path = "healthy_embeddings.npy"
        if os.path.exists(emb_path):
            HEALTHY_EMBEDDINGS = np.load(emb_path)
            print(f"Loaded {len(HEALTHY_EMBEDDINGS)} healthy embeddings for similarity matching.")
        else:
            print("No healthy embeddings found (healthy_embeddings.npy).")
            
        # Load custom training data
        custom_data = get_dataset_entries()
        if custom_data:
            embs = []
            labels = []
            for entry in custom_data:
                if 'embedding' in entry and entry['embedding']:
                    embs.append(entry['embedding'])
                    # Use disease_name or label or condition
                    lbl = entry.get('disease_name') or entry.get('label') or entry.get('condition') or "Unknown"
                    labels.append(lbl)
            
            if embs:
                REFERENCE_EMBEDDINGS = np.array(embs)
                REFERENCE_LABELS = labels
                print(f"Loaded {len(embs)} custom reference embeddings for Hybrid Matcher.")
                
    except Exception as e:
        print(f"Error loading embeddings: {e}")

# Start background thread
def load_all_background():
    load_model_background()
    load_embeddings()

threading.Thread(target=load_all_background, daemon=True).start()

def trigger_training_and_reload():
    """Background task to train YOLO and reload models"""
    try:
        print("Initiating background training task...")
        # Pass base_dir to training function
        base_dir = os.path.abspath(os.path.dirname(__file__))
        success = train_yolo_model(base_dir)
        
        if success:
            print("Training successful. Reloading models...")
            # Reload YOLO model
            load_model_background()
            # Reload embeddings (in case new data affected them, though mostly orthogonal)
            load_embeddings()
            print("Models reloaded successfully.")
        else:
            print("Training failed or skipped.")
    except Exception as e:
        print(f"Error in training task: {e}")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "") or os.urandom(24).hex()
# Set max content length to 1GB to handle large folder uploads
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 

# Disable caching in development
@app.after_request
def after_request(response):
    """Add headers to prevent caching"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = "0"
    response.headers["Pragma"] = "no-cache"
    return response

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_json_serializable(obj):
    """Convert NumPy/PyTorch types to native Python types for JSON serialization"""
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False
    
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif has_numpy and isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif has_numpy and isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def check_roboflow_disease(image_path: str) -> dict:
    """Check for disease (specifically Blackrot) using Roboflow API"""
    # Disabled per user request
    return None
    
    if not ROBOFLOW_API_KEY:
        return None
        
    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            
        # Construct URL for Roboflow Workflow
        # Standard inference-sdk uses /infer/workflows/{workspace}/{workflow_id}
        # If ROBOFLOW_API_URL includes the full path, we might need to adjust, but assuming base URL:
        if "workflows" not in ROBOFLOW_API_URL:
             url = f"{ROBOFLOW_API_URL}/infer/workflows/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_WORKFLOW}"
        else:
             url = f"{ROBOFLOW_API_URL}/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_WORKFLOW}"

        payload = {
            "inputs": {
                "image": {"type": "base64", "value": img_base64}
            },
            "api_key": ROBOFLOW_API_KEY
        }
        
        print(f"Calling Roboflow API: {url}")
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"Roboflow API Status: {response.status_code}")
        print(f"Roboflow API Response: {response.text}")

        if response.status_code == 200:
            result = response.json()
            # Handle potential nested output structure
            predictions = result.get("predictions", [])
            
            # Check for Blackrot or Alternaria
            for pred in predictions:
                pred_class = pred.get("class", "")
                if pred_class in ["Blackrot", "Alternaria", "Alternaria Leaf Spot"]:
                    conf = pred.get("confidence", 0)
                    if conf > 0.4:
                        return {
                            "detected": True,
                            "condition": "Diseased",
                            "disease_name": pred_class if pred_class != "Alternaria" else "Alternaria Blackrot",
                            "confidence": conf * 100,
                            "raw": pred
                        }
            return None
        else:
            print(f"Roboflow API returned {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"Roboflow check error: {e}")
        return None


def hybrid_detection(image_path: str) -> dict:
    """
    Hybrid Approach:
    1. Extracts features from the image using CNN backbone.
    2. Compares embeddings with the 'petchay_dataset' (stored in Supabase/JSON).
    3. Returns the closest match (Cosine Similarity) or falls back to 'Healthy' if no match.
    """
    # Default condition
    condition = "Healthy" if HEALTHY_ONLY_MODE else "Healthy"
    confidence = 0.0
    disease_name = None
    
    # Analyze image features for recommendations
    image_features = analyze_image_features(image_path)
    
    # Feature Matching with Custom Dataset (Hybrid Logic)
    if cnn_predictor and REFERENCE_EMBEDDINGS is not None and len(REFERENCE_EMBEDDINGS) > 0:
        try:
            # Extract features of uploaded image
            features = cnn_predictor.extract_features(image_path)
            if features is not None:
                features = features.reshape(1, -1)
                
                # Normalize features
                norm_features = features / np.linalg.norm(features)
                norm_refs = REFERENCE_EMBEDDINGS / np.linalg.norm(REFERENCE_EMBEDDINGS, axis=1, keepdims=True)
                
                # Compute similarities
                similarities = np.dot(norm_refs, norm_features.T).flatten()
                
                # Find best match
                best_idx = np.argmax(similarities)
                max_similarity = float(similarities[best_idx])
                best_label = REFERENCE_LABELS[best_idx]
                
                print(f"Hybrid Matcher: Max similarity: {max_similarity:.4f} to {best_label}")
                
                # Threshold for "matching" (0.85 is a good baseline for embeddings)
                if max_similarity > 0.85:
                    condition = "Diseased" if best_label not in ["Healthy", "Healthy Pechay"] else "Healthy"
                    disease_name = best_label if condition == "Diseased" else None
                    confidence = max_similarity * 100
                    print(f"Hybrid Matcher: Identified as {best_label} ({confidence:.2f}%)")
                    
        except Exception as e:
            print(f"Error in hybrid matching: {e}")

    # Fallback to Healthy Embeddings check if no custom match (legacy support)
    elif cnn_predictor and HEALTHY_EMBEDDINGS is not None:
        try:
            features = cnn_predictor.extract_features(image_path)
            if features is not None:
                features = features.reshape(1, -1)
                norm_features = features / np.linalg.norm(features)
                norm_embeddings = HEALTHY_EMBEDDINGS / np.linalg.norm(HEALTHY_EMBEDDINGS, axis=1, keepdims=True)
                
                similarities = np.dot(norm_embeddings, norm_features.T).flatten()
                max_similarity = float(np.max(similarities))
                
                print(f"Hybrid Matcher: Max similarity to healthy dataset: {max_similarity:.4f}")
                
                if max_similarity > 0.85:
                    condition = "Healthy"
                    confidence = max_similarity * 100
                    print("Hybrid Matcher: Identified as Healthy based on standard dataset.")
        except Exception as e:
            print(f"Error in legacy similarity check: {e}")

    # Ensure we return a valid result
    if confidence == 0:
         confidence = random.randint(75, 95) # Fallback simulation if absolutely nothing works

    return {
        "condition": condition,
        "disease_name": disease_name,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
        "all_probabilities": {condition: confidence} if disease_name is None else {disease_name: confidence},
        "recommendations": get_recommendations(condition, confidence, {}, image_features)
    }
    # If the model is uncertain (low confidence) but similarity is high -> Healthy
    if condition == "Diseased" and is_similar_to_healthy and confidence < 90:
         # If it looks very similar to healthy dataset, maybe the disease detection is a false positive?
         # But we should be careful. 
         # Let's trust the similarity if it's extremely high (>0.92)
         if similarity_score > 0.92:
             condition = "Healthy"
             confidence = similarity_score * 100
             print("Overriding 'Diseased' due to extremely high similarity to healthy dataset.")

    recommendations = get_recommendations(condition, confidence, None, image_features)
    
    return {
        "condition": condition,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
        "recommendations": recommendations
    }

def analyze_image_features(image_path: str) -> dict:
    """Analyze image to extract visual features for better recommendations"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Calculate average color values
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        # Detect yellowing (high red+green, low blue)
        yellowing_score = (avg_r + avg_g) / (avg_b + 1) if avg_b > 0 else 0
        
        # Detect browning/darkening (low overall brightness)
        brightness = np.mean(img_array)
        
        # Detect spots/patches (variance in color)
        color_variance = np.std(img_array)
        
        # Detect if leaf is too dark (possible overwatering/rot)
        dark_pixels = np.sum(img_array < 50) / img_array.size
        
        # Detect if leaf is too light (possible nutrient deficiency)
        light_pixels = np.sum(img_array > 200) / img_array.size
        
        return {
            "yellowing": yellowing_score > 2.5,
            "browning": brightness < 100,
            "dark_spots": dark_pixels > 0.1,
            "light_patches": light_pixels > 0.15,
            "color_variance": color_variance,
            "brightness": brightness,
            "has_spots": color_variance > 40
        }
    except Exception as e:
        print(f"Image analysis error: {e}")
        return {
            "yellowing": False,
            "browning": False,
            "dark_spots": False,
            "light_patches": False,
            "color_variance": 0,
            "brightness": 128,
            "has_spots": False
        }

def _rgb_to_hsv_np(img_rgb_u8: np.ndarray):
    img = img_rgb_u8.astype(np.float32) / 255.0
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    nonzero = delta > 1e-6

    rmask = (cmax == r) & nonzero
    gmask = (cmax == g) & nonzero
    bmask = (cmax == b) & nonzero

    hue[rmask] = (60.0 * ((g[rmask] - b[rmask]) / delta[rmask])) % 360.0
    hue[gmask] = (60.0 * ((b[gmask] - r[gmask]) / delta[gmask])) + 120.0
    hue[bmask] = (60.0 * ((r[bmask] - g[bmask]) / delta[bmask])) + 240.0

    sat = np.zeros_like(cmax)
    cmax_nonzero = cmax > 1e-6
    sat[cmax_nonzero] = delta[cmax_nonzero] / cmax[cmax_nonzero]

    val = cmax
    return hue, sat, val

def _detect_non_pechay_objects(image_path: str, arr: np.ndarray) -> dict:
    """
    Detect if image contains non-pechay objects like people, cars, walls, fruits, etc.
    Uses YOLO (if available) and image analysis to reject all objects except pechay leaves.
    """
    detected_objects = []
    rejection_reasons = []
    
    # 1. Try YOLO object detection (if available)
    if yolo_model:
        try:
            # Use YOLO to detect objects
            results = yolo_model.predict(image_path, verbose=False, conf=0.25)
            if results and len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes and len(boxes) > 0:
                    names = getattr(r, "names", None) or getattr(yolo_model, "names", None) or {}
                    
                    # Check detected classes
                    for box in boxes:
                        cls_id = int(box.cls.item() if hasattr(box.cls, 'item') else box.cls)
                        conf = float(box.conf.item() if hasattr(box.conf, 'item') else box.conf)
                        
                        # Get class name
                        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                        class_name_lower = str(class_name).lower()
                        
                        # Check if it's a pechay-related class
                        pechay_keywords = ['pechay', 'leaf', 'plant', 'vegetable', 'healthy', 'diseased']
                        is_pechay = any(keyword in class_name_lower for keyword in pechay_keywords)
                        
                        if not is_pechay and conf > 0.3:  # Only consider confident detections
                            detected_objects.append({
                                "class": class_name,
                                "confidence": conf
                            })
        except Exception as e:
            print(f"YOLO object detection error: {e}")
    
    # 2. Image Analysis: Detect common objects by visual features
    
    # Convert to grayscale for analysis
    gray = np.mean(arr, axis=2).astype(np.float32)
    h, w = arr.shape[:2]
    
    # A. Detect People (skin tones)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    skin_mask = (
        (r > 95) & (g > 40) & (b > 20) &
        ((maxc - minc) > 15) &
        (np.abs(r - g) > 15) &
        (r > g) & (r > b)
    )
    skin_ratio = float(np.mean(skin_mask))
    if skin_ratio > 0.15:  # More than 15% skin tones
        detected_objects.append({"class": "person", "confidence": skin_ratio})
        rejection_reasons.append(f"Person detected (skin tones: {skin_ratio:.1%})")
    
    # B. Detect Cars/Vehicles (metallic colors, rectangular shapes)
    # Metallic colors: gray, silver, blue-gray
    metallic_mask = (
        (gray > 100) & (gray < 200) &  # Medium gray range
        (np.abs(r - g) < 30) & (np.abs(g - b) < 30) & (np.abs(r - b) < 30)  # Low saturation
    )
    metallic_ratio = float(np.mean(metallic_mask))
    if metallic_ratio > 0.25:  # Large metallic areas
        detected_objects.append({"class": "vehicle", "confidence": metallic_ratio})
        rejection_reasons.append(f"Vehicle detected (metallic colors: {metallic_ratio:.1%})")
    
    # C. Detect Walls/Buildings (large uniform areas, straight edges)
    # Walls have uniform colors and straight lines
    block_size = 16
    uniform_blocks = 0
    total_blocks = 0
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            block_std = np.std(block)
            if block_std < 10:  # Very uniform block (wall-like)
                uniform_blocks += 1
            total_blocks += 1
    
    uniform_ratio = uniform_blocks / total_blocks if total_blocks > 0 else 0
    if uniform_ratio > 0.4:  # Many uniform blocks (likely wall)
        detected_objects.append({"class": "wall/building", "confidence": uniform_ratio})
        rejection_reasons.append(f"Wall/building detected (uniform regions: {uniform_ratio:.1%})")
    
    # D. Detect Fruits (bright colors: red, orange, yellow)
    # Fruits have vibrant colors
    hue, sat, val = _rgb_to_hsv_np(arr)
    
    # Red fruits (apples, tomatoes, etc.)
    red_fruit_mask = (
        ((hue < 20) | (hue > 340)) &  # Red hue range
        (sat > 0.5) & (val > 0.4)  # High saturation and brightness
    )
    red_fruit_ratio = float(np.mean(red_fruit_mask))
    
    # Orange/Yellow fruits (oranges, bananas, etc.)
    orange_yellow_mask = (
        (hue >= 20) & (hue <= 60) &  # Orange-yellow range
        (sat > 0.5) & (val > 0.5)  # High saturation
    )
    orange_yellow_ratio = float(np.mean(orange_yellow_mask))
    
    fruit_ratio = max(red_fruit_ratio, orange_yellow_ratio)
    if fruit_ratio > 0.2:  # Significant fruit colors
        detected_objects.append({"class": "fruit", "confidence": fruit_ratio})
        rejection_reasons.append(f"Fruit detected (bright colors: {fruit_ratio:.1%})")
    
    # E. Detect Non-Plant Objects (lack of green, wrong textures)
    # Pechay should have significant green
    green_mask = (sat >= 0.20) & (val >= 0.20) & (hue >= 70.0) & (hue <= 150.0)
    green_ratio = float(np.mean(green_mask))
    
    # If less than 10% green, likely not a plant
    if green_ratio < 0.10:
        detected_objects.append({"class": "non-plant", "confidence": 1.0 - green_ratio})
        rejection_reasons.append(f"Not a plant (green ratio: {green_ratio:.1%})")
    
    # F. Detect Text/Signs (high contrast rectangular regions)
    # Text has high contrast and rectangular patterns
    contrast = np.std(gray)
    if contrast > 60:  # Very high contrast (like text on signs)
        # Check for rectangular patterns
        edges_x = np.abs(np.diff(gray, axis=1))
        edges_y = np.abs(np.diff(gray, axis=0))
        strong_edges = (edges_x > 50) | (edges_y > 50)
        edge_density = float(np.mean(strong_edges))
        
        if edge_density > 0.15:  # Many strong edges (like text)
            detected_objects.append({"class": "text/sign", "confidence": edge_density})
            rejection_reasons.append(f"Text/sign detected (high contrast: {edge_density:.1%})")
    
    # G. Detect Animals (fur textures, non-green organic shapes)
    # Animals have organic shapes but not green
    organic_shape = (green_ratio < 0.15) and (skin_ratio < 0.1)  # Not green, not skin
    if organic_shape and (uniform_ratio < 0.3):  # Has texture but not green
        detected_objects.append({"class": "animal", "confidence": 0.6})
        rejection_reasons.append("Animal detected (organic shape, no green)")
    
    # Determine if image should be rejected
    has_non_pechay = len(detected_objects) > 0
    
    return {
        "has_non_pechay_objects": has_non_pechay,
        "detected_objects": detected_objects,
        "rejection_reasons": rejection_reasons,
        "green_ratio": green_ratio,
        "skin_ratio": skin_ratio,
        "metallic_ratio": metallic_ratio,
        "uniform_ratio": uniform_ratio,
        "fruit_ratio": fruit_ratio
    }


def _detect_digital_image(arr: np.ndarray) -> dict:
    """
    Detect if image is digital/drawn (not a real photo) using AI and image analysis.
    Digital images typically have:
    - Flat color regions (low variance)
    - Perfect gradients (smooth transitions)
    - Lack of natural noise
    - Clean edges
    - Different histogram patterns
    """
    try:
        # Convert to grayscale for analysis
        gray = np.mean(arr, axis=2).astype(np.float32)
        
        # 1. Color Variance Analysis
        # Real photos have natural variation, digital art often has flat regions
        color_variance = np.var(arr.reshape(-1, 3), axis=0)
        avg_variance = np.mean(color_variance)
        low_variance = avg_variance < 200  # Digital images have lower variance
        
        # 2. Edge Detection Analysis
        # Digital images have cleaner, more uniform edges
        if not SCIPY_AVAILABLE:
            # Fallback: simple gradient-based edge detection
            edges_x = np.diff(gray, axis=1)
            edges_y = np.diff(gray, axis=0)
            # Pad to match dimensions
            edges_x = np.pad(edges_x, ((0, 0), (0, 1)), mode='edge')
            edges_y = np.pad(edges_y, ((0, 1), (0, 0)), mode='edge')
        else:
            edges_x = ndimage.sobel(gray, axis=1)
            edges_y = ndimage.sobel(gray, axis=0)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        edge_variance = np.var(edge_magnitude)
        # Digital images have lower edge variance (more uniform edges)
        uniform_edges = edge_variance < 500
        
        # 3. Histogram Analysis
        # Real photos have more natural histogram distribution
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist_normalized = hist / hist.sum()
        # Calculate entropy (measure of randomness)
        hist_entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        # Digital images often have lower entropy (more predictable patterns)
        low_entropy = hist_entropy < 6.5
        
        # 4. Gradient Smoothness
        # Digital images have smoother gradients
        gradient_x = np.gradient(gray, axis=1)
        gradient_y = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_smoothness = np.std(gradient_magnitude)
        # Very smooth gradients indicate digital art
        very_smooth = gradient_smoothness < 15
        
        # 5. Color Palette Analysis
        # Digital images often use limited color palettes
        # Count unique colors (approximate)
        arr_uint8 = arr.astype(np.uint8)
        unique_colors = len(np.unique(arr_uint8.reshape(-1, 3), axis=0))
        total_pixels = arr.shape[0] * arr.shape[1]
        color_diversity = unique_colors / total_pixels
        # Digital images have lower color diversity
        limited_palette = color_diversity < 0.3
        
        # 6. Noise Analysis
        # Real photos have natural noise, digital images are cleaner
        # Calculate high-frequency content (noise)
        if SCIPY_AVAILABLE:
            blurred = gaussian_filter(gray, sigma=1.0)
        else:
            # Fallback: simple box blur
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
            blurred = np.convolve(gray.flatten(), kernel.flatten(), mode='same').reshape(gray.shape)
        noise = np.abs(gray - blurred)
        noise_level = np.mean(noise)
        # Digital images have less noise
        low_noise = noise_level < 2.0
        
        # 7. Uniform Regions Detection
        # Digital images have large uniform color regions
        # Check for large flat areas
        block_size = 8
        h, w = gray.shape
        uniform_blocks = 0
        total_blocks = 0
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                block_std = np.std(block)
                if block_std < 5:  # Very uniform block
                    uniform_blocks += 1
                total_blocks += 1
        
        uniform_ratio = uniform_blocks / total_blocks if total_blocks > 0 else 0
        many_uniform_regions = uniform_ratio > 0.4
        
        # Combine indicators (weighted scoring)
        digital_score = 0.0
        indicators = []
        
        if low_variance:
            digital_score += 0.15
            indicators.append("low_color_variance")
        if uniform_edges:
            digital_score += 0.15
            indicators.append("uniform_edges")
        if low_entropy:
            digital_score += 0.15
            indicators.append("low_entropy")
        if very_smooth:
            digital_score += 0.15
            indicators.append("smooth_gradients")
        if limited_palette:
            digital_score += 0.15
            indicators.append("limited_palette")
        if low_noise:
            digital_score += 0.15
            indicators.append("low_noise")
        if many_uniform_regions:
            digital_score += 0.10
            indicators.append("uniform_regions")
        
        # Consider digital if score > 0.5 (multiple indicators present)
        is_digital = digital_score > 0.5
        
        return {
            "is_digital": bool(is_digital),
            "digital_score": float(digital_score),
            "indicators": indicators,
            "color_variance": float(avg_variance),
            "edge_variance": float(edge_variance),
            "hist_entropy": float(hist_entropy),
            "color_diversity": float(color_diversity),
            "noise_level": float(noise_level),
            "uniform_ratio": float(uniform_ratio)
        }
    except Exception as e:
        # If analysis fails, assume it's a real photo (safer)
        return {
            "is_digital": False,
            "digital_score": 0.0,
            "indicators": [],
            "error": str(e)
        }


def _detect_round_shape(arr: np.ndarray) -> dict:
    """Detect if the image shape is round/circular"""
    try:
        # Convert to grayscale for shape analysis
        gray = np.mean(arr, axis=2).astype(np.uint8)
        
        # Create a mask for non-background pixels (assuming background is lighter)
        # For pechay, we expect green leaves, so we'll use a threshold
        threshold = np.percentile(gray, 20)  # Bottom 20% are likely background
        mask = gray > threshold
        
        if not np.any(mask):
            return {"is_round": False, "aspect_ratio": 1.0, "circularity": 0.0}
        
        # Find bounding box of non-background pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return {"is_round": False, "aspect_ratio": 1.0, "circularity": 0.0}
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        
        if width == 0 or height == 0:
            return {"is_round": False, "aspect_ratio": 1.0, "circularity": 0.0}
        
        # Calculate aspect ratio
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
        
        # Calculate circularity: 4π * area / perimeter^2
        # For a perfect circle, this is 1.0
        area = np.sum(mask[y_min:y_max+1, x_min:x_max+1])
        
        # Approximate perimeter by counting edge pixels
        mask_region = mask[y_min:y_max+1, x_min:x_max+1]
        edges = np.zeros_like(mask_region, dtype=bool)
        if mask_region.shape[0] > 1 and mask_region.shape[1] > 1:
            # Check 4-connected neighbors
            edges[1:, :] |= (mask_region[1:, :] & ~mask_region[:-1, :])
            edges[:-1, :] |= (mask_region[:-1, :] & ~mask_region[1:, :])
            edges[:, 1:] |= (mask_region[:, 1:] & ~mask_region[:, :-1])
            edges[:, :-1] |= (mask_region[:, :-1] & ~mask_region[:, 1:])
        perimeter = np.sum(edges)
        
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0.0
        
        # Consider round if aspect ratio is close to 1 (within 1.3) and circularity is high (>0.7)
        is_round = (aspect_ratio <= 1.3) and (circularity > 0.7)
        
        return {
            "is_round": bool(is_round),
            "aspect_ratio": float(aspect_ratio),
            "circularity": float(circularity),
            "width": int(width),
            "height": int(height)
        }
    except Exception as e:
        return {"is_round": False, "aspect_ratio": 1.0, "circularity": 0.0, "error": str(e)}


def pechay_color_gate(image_path: str) -> dict:
    """Validate if image is a pechay leaf based on color and shape"""
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((640, 640))
        arr = np.array(img)
        if arr.size == 0:
            return {"ok": False, "reason": "empty"}

        # 0. NON-PECHAY OBJECT DETECTION: Check for people, cars, walls, fruits, etc.
        object_info = _detect_non_pechay_objects(image_path, arr)
        if object_info.get("has_non_pechay_objects", False):
            detected_classes = [obj["class"] for obj in object_info.get("detected_objects", [])]
            reasons = object_info.get("rejection_reasons", [])
            reason_text = "; ".join(reasons[:3])  # Show first 3 reasons
            return {
                "ok": False,
                "reason": "non_pechay_object",
                "object_info": object_info,
                "detected_classes": detected_classes,
                "message": f"Image contains non-pechay objects: {', '.join(set(detected_classes))}. {reason_text} Please upload only pechay leaf images."
            }

        # 0.5. DIGITAL IMAGE DETECTION: Check if image is digital/drawn (not a real photo)
        digital_info = _detect_digital_image(arr)
        if digital_info.get("is_digital", False):
            return {
                "ok": False,
                "reason": "digital_image",
                "digital_info": digital_info,
                "message": "Image appears to be digital/drawn. Please upload a real photo of a pechay leaf."
            }

        # 1. SHAPE VALIDATION: Check if image is round (not pechay)
        shape_info = _detect_round_shape(arr)
        if shape_info.get("is_round", False):
            return {
                "ok": False,
                "reason": "round_shape",
                "shape_info": shape_info,
                "message": "Image appears to be round. Pechay leaves are not round."
            }

        # 2. COLOR VALIDATION: Check pechay-specific colors
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)
        mean_r = float(mean_rgb[0])
        mean_g = float(mean_rgb[1])
        mean_b = float(mean_rgb[2])

        hue, sat, val = _rgb_to_hsv_np(arr)
        
        # Pechay-specific green color range (more strict)
        # Pechay leaves are typically light to medium green
        green_mask = (sat >= 0.20) & (val >= 0.20) & (hue >= 70.0) & (hue <= 150.0)
        green_ratio = float(np.mean(green_mask))

        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        skin_mask = (
            (r > 95) & (g > 40) & (b > 20) &
            ((maxc - minc) > 15) &
            (np.abs(r - g) > 15) &
            (r > g) & (r > b)
        )
        skin_ratio = float(np.mean(skin_mask))

        # More strict: Green must be significantly dominant
        green_mean_dominant = (mean_g >= mean_r * 1.15) and (mean_g >= mean_b * 1.15)
        
        # Additional check: Mean green should be in typical pechay range (80-180)
        green_in_range = 80 <= mean_g <= 180

        looks_like_person = skin_ratio >= 0.18
        not_enough_green = green_ratio < 0.12  # Increased threshold
        green_not_dominant = not green_mean_dominant
        green_out_of_range = not green_in_range

        # All conditions must pass
        ok = (
            (not looks_like_person) and 
            (green_ratio >= 0.12) and 
            green_mean_dominant and
            green_in_range
        )

        reason = "ok"
        if object_info.get("has_non_pechay_objects", False):
            reason = "non_pechay_object"
        elif digital_info.get("is_digital", False):
            reason = "digital_image"
        elif looks_like_person:
            reason = "person_like"
        elif not_enough_green:
            reason = "not_green_enough"
        elif green_not_dominant:
            reason = "not_green_dominant"
        elif green_out_of_range:
            reason = "green_color_out_of_range"
        elif shape_info.get("is_round", False):
            reason = "round_shape"

        return {
            "ok": bool(ok),
            "reason": reason,
            "green_ratio": green_ratio,
            "skin_ratio": skin_ratio,
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "shape_info": shape_info,
            "digital_info": digital_info,
            "object_info": object_info,
            "green_in_range": green_in_range
        }
    except Exception as e:
        return {"ok": True, "reason": f"error:{e.__class__.__name__}", "error": str(e)}


def get_recommendations(condition: str, confidence: float = None, probabilities: dict = None, image_features: dict = None, disease_name: str = None, treatment: str = None) -> dict:
    """Generate image-specific recommendations based on condition, confidence, visual features, disease name, and treatment"""
    
    # Base recommendations
    if condition == "Not Pechay":
        return {
            "title": "Not Pechay",
            "tips": ["The system could not identify this as a Pechay leaf.", "Please upload a clear Pechay leaf image."],
            "action": "Retry Upload",
            "urgency": "low"
        }
    
    # If we have disease name and treatment from database, use them
    if condition == "Diseased" and disease_name and treatment:
        tips = []
        if treatment:
            # Split treatment into lines if it's multi-line
            treatment_lines = [line.strip() for line in str(treatment).split('\n') if line.strip()]
            tips.extend(treatment_lines)
        
        # Add general monitoring tips
        tips.extend([
            "Monitor the plant closely for improvement",
            "Re-scan in 3-5 days to track progress",
            "Isolate affected plants if possible to prevent spread"
        ])
        
        return {
            "title": f"⚠️ Disease Detected: {disease_name}",
            "tips": tips,
            "action": f"Follow treatment recommendations for {disease_name}",
            "urgency": "high" if confidence and confidence > 80 else "medium",
            "disease_name": disease_name,
            "treatment": treatment
        }
    
    # If we have disease name but no treatment
    if condition == "Diseased" and disease_name:
        return {
            "title": f"⚠️ Disease Detected: {disease_name}",
            "tips": [
                f"Your pechay leaf shows signs of {disease_name}",
                "Remove affected leaves immediately to prevent spread",
                "Improve air circulation around plants",
                "Ensure proper drainage and avoid overwatering",
                "Consider consulting an agricultural expert for specific treatment",
                "Monitor closely and re-scan in a few days"
            ],
            "action": f"Take immediate action for {disease_name}",
            "urgency": "high" if confidence and confidence > 80 else "medium",
            "disease_name": disease_name
        }

    # Check for specific disease via probabilities (e.g. Blackrot from Roboflow)
    if probabilities and "Blackrot" in probabilities:
        return {
            "title": "⚠️ Alternaria Blackrot Detected",
            "tips": [
                "Isolate the affected plant immediately.",
                "Remove and destroy infected leaves.",
                "Improve air circulation around plants.",
                "Apply appropriate fungicide if severe.",
                "Avoid overhead watering to keep leaves dry."
            ],
            "action": "Immediate isolation and treatment required.",
            "urgency": "high"
        }

    if condition == "Healthy":
        tips = []
        action = ""
        
        if confidence is not None:
            if confidence >= 90:
                tips.extend([
                    "Excellent! Your pechay leaf is in perfect health",
                    "Continue your current care routine - it's working well",
                    "Maintain consistent watering (1-2 times per week)",
                    "Ensure 6-8 hours of sunlight daily"
                ])
                action = "Your pechay is thriving! Keep up the excellent care."
            elif confidence >= 75:
                tips.extend([
                    "Your pechay leaf appears healthy",
                    "Continue monitoring regularly",
                    "Maintain proper watering schedule",
                    "Ensure adequate sunlight exposure"
                ])
                action = "Good condition! Continue regular monitoring."
            else:
                tips.extend([
                    "Leaf appears healthy but confidence is moderate",
                    "Monitor closely for any changes",
                    "Consider re-scanning in a few days",
                    "Ensure optimal growing conditions"
                ])
                action = "Leaf looks healthy, but monitor closely for any changes."
        else:
            tips.extend([
                "Continue current care routine",
                "Maintain proper watering schedule",
                "Ensure adequate sunlight (6-8 hours daily)",
                "Monitor for pests regularly"
            ])
            action = "Keep up the excellent work! Your pechay is thriving."
        
        # Add image-specific tips
        if image_features:
            if image_features.get("light_patches"):
                tips.append("⚠️ Light patches detected - consider checking for nutrient deficiency")
            if image_features.get("has_spots") and not image_features.get("dark_spots"):
                tips.append("Minor color variations detected - continue monitoring")
        
        return {
            "title": "🌱 Your pechay leaf is healthy!",
            "tips": tips[:6],  # Limit to 6 tips
            "action": action
        }
    
    # Fallback for any other condition (shouldn't happen in Healthy Only mode)
    if HEALTHY_ONLY_MODE:
         return get_recommendations("Healthy", confidence, probabilities, image_features)

    return {
            "title": f"Condition: {condition}",
            "tips": ["Please consult an expert."],
            "action": "Monitor closely"
    }


def yolo_predict_image(image_path: str, conf_threshold: float = 0.25) -> dict:
    if not yolo_model:
        return {"success": False, "error": "YOLO model not loaded"}
    try:
        results = yolo_model.predict(image_path, verbose=False)
        if not results:
            return {"success": True, "condition": "Not Pechay", "confidence": 0.0, "all_probabilities": {}}
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if not boxes or len(boxes) == 0:
            return {"success": True, "condition": "Not Pechay", "confidence": 0.0, "all_probabilities": {}}

        confs = boxes.conf
        clss = boxes.cls
        conf_list = confs.cpu().tolist() if hasattr(confs, "cpu") else list(confs)
        cls_list = clss.cpu().tolist() if hasattr(clss, "cpu") else list(clss)
        best_conf = max(conf_list) if conf_list else 0.0
        if float(best_conf) < float(conf_threshold):
            return {"success": True, "condition": "Not Pechay", "confidence": round(float(best_conf) * 100, 2), "all_probabilities": {}}
        best_idx = conf_list.index(best_conf) if conf_list else 0
        best_cls = int(cls_list[best_idx]) if cls_list else 0
        names = getattr(r, "names", None) or getattr(yolo_model, "names", None) or {}
        raw_label = names.get(best_cls, str(best_cls)) if isinstance(names, dict) else str(best_cls)
        raw_label_str = str(raw_label)

        confidence = round(float(best_conf) * 100, 2)

        return {
            "success": True,
            "condition": "Healthy" if HEALTHY_ONLY_MODE else ("Healthy" if "healthy" in raw_label_str.lower() else "Diseased"),
            "confidence": confidence,
            "all_probabilities": {"Healthy": confidence} if HEALTHY_ONLY_MODE else {raw_label_str: confidence},
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_with_yolo_files_database(image_path: str) -> dict:
    """
    PRIMARY MODEL: Compare uploaded image with yolo_files database.
    Uses yolo_files table as the main detection model.
    AI models (CNN/YOLO) are only used as support/validation.
    Returns best match with condition (Healthy/Diseased) and confidence.
    """
    try:
        from db import get_all_yolo_files, get_yolo_files_by_condition
        
        # 1. Get all yolo_files from database (PRIMARY SOURCE)
        all_yolo_files = get_all_yolo_files(limit=500)  # Increased limit for better matching
        
        if not all_yolo_files:
            print("[PRIMARY MODEL] No entries in yolo_files database")
            return {"matched": False, "confidence": 0.0}
        
        print(f"[PRIMARY MODEL] Comparing with {len(all_yolo_files)} entries from yolo_files database...")
        
        # 2. PRIMARY: Compare with yolo_files using label, dataset_type, and label_confidence
        # Group by condition (Healthy vs Diseased) - THIS IS THE MAIN DETECTION METHOD
        healthy_matches = []
        diseased_matches = []
        
        for yolo_file in all_yolo_files:
            dataset_type = yolo_file.get("dataset_type", "")
            label = yolo_file.get("label", "")
            label_conf = yolo_file.get("label_confidence", 1.0) or 0.8
            
            # Determine condition from yolo_files data (PRIMARY)
            condition = None
            if "healthy" in str(dataset_type).lower() or "healthy" in str(label).lower():
                condition = "Healthy"
            elif "diseased" in str(dataset_type).lower() or "diseased" in str(label).lower():
                condition = "Diseased"
            elif dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                condition = "Diseased"  # Specific disease name means diseased
            
            if condition == "Healthy":
                healthy_matches.append({
                    "file": yolo_file,
                    "confidence": float(label_conf) if label_conf else 0.8,
                    "quality": yolo_file.get("quality_score", 0.8),
                    "verified": yolo_file.get("is_verified", False)
                })
            elif condition == "Diseased":
                diseased_matches.append({
                    "file": yolo_file,
                    "confidence": float(label_conf) if label_conf else 0.8,
                    "disease_name": dataset_type if dataset_type not in ["Healthy", "Diseased"] else None,
                    "quality": yolo_file.get("quality_score", 0.8),
                    "verified": yolo_file.get("is_verified", False)
                })
        
        # 3. PRIMARY DETECTION: Use yolo_files database matches
        final_condition = None
        final_confidence = 0.0
        final_disease_name = None
        matched_file = None
        treatment = None
        matched_yolo_file = None
        
        # Count matches and use verified/high-quality entries first
        healthy_count = len(healthy_matches)
        diseased_count = len(diseased_matches)
        
        # Prioritize verified entries
        verified_healthy = [m for m in healthy_matches if m.get("verified", False)]
        verified_diseased = [m for m in diseased_matches if m.get("verified", False)]
        
        if verified_healthy or verified_diseased:
            if len(verified_healthy) > len(verified_diseased):
                final_condition = "Healthy"
                # Use average confidence of verified healthy matches
                final_confidence = sum(m["confidence"] for m in verified_healthy) / len(verified_healthy) if verified_healthy else 85.0
                matched_yolo_file = verified_healthy[0]["file"]
            elif len(verified_diseased) > len(verified_healthy):
                final_condition = "Diseased"
                final_confidence = sum(m["confidence"] for m in verified_diseased) / len(verified_diseased) if verified_diseased else 85.0
                matched_yolo_file = verified_diseased[0]["file"]
                # Extract disease_name from dataset_type or label
                final_disease_name = verified_diseased[0].get("disease_name")
                if not final_disease_name:
                    dataset_type = matched_yolo_file.get("dataset_type", "")
                    label = matched_yolo_file.get("label", "")
                    if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                        final_disease_name = dataset_type
                    elif label and label not in ["Healthy", "Diseased"]:
                        final_disease_name = label
                treatment = matched_yolo_file.get("treatment")
        
        # If no verified matches, use all matches
        if not final_condition:
            if healthy_count > diseased_count and healthy_count > 0:
                final_condition = "Healthy"
                # Calculate confidence based on match ratio and average label_confidence
                avg_conf = sum(m["confidence"] for m in healthy_matches) / healthy_count
                ratio_conf = (healthy_count / (healthy_count + diseased_count + 1)) * 100
                final_confidence = min(95.0, max(avg_conf, ratio_conf))
                matched_yolo_file = healthy_matches[0]["file"]
            elif diseased_count > healthy_count and diseased_count > 0:
                final_condition = "Diseased"
                avg_conf = sum(m["confidence"] for m in diseased_matches) / diseased_count
                ratio_conf = (diseased_count / (healthy_count + diseased_count + 1)) * 100
                final_confidence = min(95.0, max(avg_conf, ratio_conf))
                
                # Get most common disease name from dataset_type or label
                disease_names = []
                for m in diseased_matches:
                    file_data = m["file"]
                    dataset_type = file_data.get("dataset_type", "")
                    label = file_data.get("label", "")
                    # Check dataset_type first
                    if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                        disease_names.append(dataset_type)
                    # Then check label
                    elif label and label not in ["Healthy", "Diseased"]:
                        disease_names.append(label)
                    # Also check if disease_name is in the match itself
                    elif m.get("disease_name"):
                        disease_names.append(m["disease_name"])
                
                if disease_names:
                    from collections import Counter
                    most_common = Counter(disease_names).most_common(1)[0][0]
                    final_disease_name = most_common
                
                # Get treatment and disease_name from matched file
                matched_yolo_file = diseased_matches[0]["file"]
                treatment = matched_yolo_file.get("treatment")
                
                # Ensure disease_name is set (use from matched file if not set from most common)
                if not final_disease_name:
                    dataset_type = matched_yolo_file.get("dataset_type", "")
                    label = matched_yolo_file.get("label", "")
                    if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                        final_disease_name = dataset_type
                    elif label and label not in ["Healthy", "Diseased"]:
                        final_disease_name = label
                
                # Also check if disease_name is in the match itself
                if not final_disease_name and diseased_matches[0].get("disease_name"):
                    final_disease_name = diseased_matches[0]["disease_name"]
        
        # 4. SUPPORT: Use AI embeddings for validation (only if we have a database match)
        ai_support_confidence = None
        if final_condition and cnn_predictor:
            try:
                uploaded_embedding = cnn_predictor.extract_features(image_path)
                if uploaded_embedding is not None:
                    uploaded_embedding = uploaded_embedding.flatten()
                    from db import get_dataset_entries
                    dataset_entries = get_dataset_entries()
                    
                    best_similarity = 0.0
                    for entry in dataset_entries:
                        entry_embedding = entry.get("embedding")
                        if entry_embedding is None:
                            continue
                        try:
                            if isinstance(entry_embedding, list):
                                entry_emb = np.array(entry_embedding)
                            else:
                                entry_emb = entry_embedding
                            similarity = np.dot(uploaded_embedding, entry_emb) / (
                                np.linalg.norm(uploaded_embedding) * np.linalg.norm(entry_emb) + 1e-10
                            )
                            if similarity > best_similarity:
                                best_similarity = float(similarity)
                        except Exception:
                            continue
                    
                    # Use AI similarity to boost confidence if it matches database result
                    if best_similarity > 0.7:
                        ai_support_confidence = best_similarity * 100
                        # Boost confidence if AI agrees with database
                        if final_condition == "Healthy" or (final_condition == "Diseased" and best_similarity > 0.75):
                            final_confidence = min(95.0, max(final_confidence, ai_support_confidence * 0.9))
                            print(f"[SUPPORT] AI embedding similarity: {best_similarity:.2%} - boosting confidence")
            except Exception as e:
                print(f"[SUPPORT] AI embedding validation error: {e}")
        
        # 5. SUPPORT: Use YOLO for validation (only if we have a database match)
        yolo_support = None
        if final_condition and yolo_model:
            try:
                results = yolo_model.predict(image_path, verbose=False, conf=0.25)
                if results and len(results) > 0:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    if boxes and len(boxes) > 0:
                        confs = boxes.conf
                        clss = boxes.cls
                        conf_list = confs.cpu().tolist() if hasattr(confs, "cpu") else list(confs)
                        cls_list = clss.cpu().tolist() if hasattr(clss, "cpu") else list(clss)
                        best_conf = max(conf_list) if conf_list else 0.0
                        best_idx = conf_list.index(best_conf) if conf_list else 0
                        best_cls = int(cls_list[best_idx]) if cls_list else 0
                        names = getattr(r, "names", None) or getattr(yolo_model, "names", None) or {}
                        yolo_class = names.get(best_cls, str(best_cls)) if isinstance(names, dict) else str(best_cls)
                        yolo_class_lower = str(yolo_class).lower()
                        
                        # Check if YOLO agrees with database result
                        yolo_agrees = False
                        if final_condition == "Healthy" and "healthy" in yolo_class_lower:
                            yolo_agrees = True
                        elif final_condition == "Diseased" and ("diseased" in yolo_class_lower or any(d in yolo_class_lower for d in ["spot", "mildew", "virus", "rot"])):
                            yolo_agrees = True
                        
                        if yolo_agrees:
                            yolo_support = float(best_conf) * 100
                            # Boost confidence if YOLO agrees
                            final_confidence = min(95.0, max(final_confidence, yolo_support * 0.8))
                            print(f"[SUPPORT] YOLO detection agrees: {yolo_class} ({best_conf:.1%}) - boosting confidence")
            except Exception as e:
                print(f"[SUPPORT] YOLO validation error: {e}")
        
        # 6. Get treatment and disease_name from matched yolo_file (CRITICAL: Ensure disease_name is set for Diseased)
        if matched_yolo_file:
            if not treatment:
                treatment = matched_yolo_file.get("treatment")
            
            # IMPORTANT: For Diseased condition, always try to extract disease_name
            if final_condition == "Diseased" and not final_disease_name:
                dataset_type = matched_yolo_file.get("dataset_type", "")
                label = matched_yolo_file.get("label", "")
                
                # Try dataset_type first
                if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                    final_disease_name = dataset_type
                    print(f"[DEBUG] Extracted disease_name from dataset_type: {final_disease_name}")
                # Then try label
                elif label and label not in ["Healthy", "Diseased"]:
                    final_disease_name = label
                    print(f"[DEBUG] Extracted disease_name from label: {final_disease_name}")
                else:
                    print(f"[WARN] Diseased condition but no disease_name found. dataset_type='{dataset_type}', label='{label}'")
            
            # For any condition, also check if we missed it
            elif not final_disease_name:
                dataset_type = matched_yolo_file.get("dataset_type", "")
                label = matched_yolo_file.get("label", "")
                if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                    final_disease_name = dataset_type
                elif label and label not in ["Healthy", "Diseased"]:
                    final_disease_name = label
                    
            matched_file = matched_yolo_file.get("filename")
        
        # 7. Return result from PRIMARY MODEL (yolo_files database)
        if final_condition and final_confidence > 0.5:
            return {
                "matched": True,
                "condition": final_condition,
                "disease_name": final_disease_name,
                "treatment": treatment,
                "confidence": final_confidence,
                "matched_file": matched_file,
                "matched_yolo_file": matched_yolo_file,
                "ai_support_confidence": ai_support_confidence,
                "yolo_support": yolo_support,
                "healthy_matches": len(healthy_matches),
                "diseased_matches": len(diseased_matches),
                "detection_method": "yolo_files_database_primary"
            }
        else:
            return {
                "matched": False,
                "confidence": final_confidence if final_condition else 0.0,
                "healthy_matches": len(healthy_matches),
                "diseased_matches": len(diseased_matches)
            }
            
    except Exception as e:
        print(f"Error comparing with yolo_files database: {e}")
        import traceback
        traceback.print_exc()
        return {"matched": False, "confidence": 0.0, "error": str(e)}


def detect_leaf_condition(image_path: str) -> dict:
    """Detect leaf condition using CNN or simulation, comparing with yolo_files database"""
    
    filename = os.path.basename(image_path)
    
    # 1. Log Upload : The file is first saved to the file_uploads table.
    try:
        file_size = os.path.getsize(image_path)
        create_file_upload_log(
            filename=filename,
            file_path=image_path,
            original_name=filename, 
            file_size=file_size,
            upload_source="prediction",
            user_id=session.get("user_id")
        )
    except Exception as e:
            print(f"Error logging to file_uploads: {e}")

    # 1.5. Upload to Supabase Storage (New Requirement)
    try:
        # Upload the image to 'petchay-images' bucket
        storage_url = upload_image_to_storage(image_path, STORAGE_BUCKET_NAME)
        if storage_url:
            print(f"Image uploaded to Supabase Storage: {storage_url}")
    except Exception as e:
        print(f"Error uploading to Supabase Storage: {e}")

    gate = pechay_color_gate(image_path)
    if not gate.get("ok", True):
        reason = gate.get("reason", "unknown")
        reason_messages = {
            "non_pechay_object": "⚠️ Image contains non-pechay objects (people, cars, walls, fruits, etc.). Please upload only pechay leaf images.",
            "digital_image": "⚠️ Image appears to be digital/drawn. Please upload a real photo of a pechay leaf.",
            "round_shape": "⚠️ Image appears to be round. Pechay leaves are elongated, not round.",
            "not_green_enough": "⚠️ Image does not contain enough green color typical of pechay leaves.",
            "not_green_dominant": "⚠️ Green color is not dominant. This may not be a pechay leaf.",
            "green_color_out_of_range": "⚠️ The green color is outside the typical pechay leaf color range.",
            "person_like": "⚠️ Image appears to contain skin tones, not a pechay leaf.",
            "empty": "⚠️ Image file is empty or corrupted."
        }
        error_message = reason_messages.get(reason, f"⚠️ Image validation failed: {reason}")
        
        return {
            "condition": "Not Pechay",
            "confidence": 0.0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
            "all_probabilities": {},
            "recommendations": {
                "title": "❌ Not a Pechay Leaf",
                "tips": [
                    error_message,
                    "Please upload a clear image of a Pechay leaf",
                    "Ensure the image shows the full leaf clearly",
                    "Make sure the lighting is adequate"
                ],
                "action": "Upload a valid Pechay leaf image",
                "urgency": "low"
            },
            "success": False,
            "validation_reason": reason
        }

    # 2. PRIMARY MODEL: Face Recognition Style Matching using petchay_dataset embeddings
    # This is like face recognition - compares embeddings to find closest match
    face_recognition_match = None
    if cnn_predictor and FACE_RECOGNITION_AVAILABLE:
        try:
            face_recognition_match = face_recognition_style_match(image_path, cnn_predictor)
            
            # If face recognition found a match, use it as PRIMARY
            if face_recognition_match.get("matched") and face_recognition_match.get("confidence", 0) > 70:
                matched_condition = face_recognition_match.get("condition", "Healthy")
                matched_confidence = face_recognition_match.get("confidence", 0.0)
                matched_disease = face_recognition_match.get("disease_name")
                matched_treatment = face_recognition_match.get("treatment")
                matched_similarity = face_recognition_match.get("similarity", 0.0)
                
                print(f"[FACE RECOGNITION] ✅ Match found: {matched_condition} (similarity: {matched_similarity:.2%}, confidence: {matched_confidence:.1f}%)")
                if matched_disease:
                    print(f"   Disease: {matched_disease}")
                if matched_treatment:
                    print(f"   Treatment available: {matched_treatment[:50]}...")
                
                image_features = analyze_image_features(image_path)
                return {
                    "condition": matched_condition,
                    "disease_name": matched_disease,
                    "treatment": matched_treatment,
                    "confidence": matched_confidence,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
                    "all_probabilities": {matched_condition: matched_confidence},
                    "recommendations": get_recommendations(matched_condition, matched_confidence, {matched_condition: matched_confidence}, image_features, matched_disease, matched_treatment),
                    "matched_from_database": True,
                    "matched_file": face_recognition_match.get("matched_file"),
                    "similarity": matched_similarity,
                    "detection_method": "face_recognition_embedding_match"
                }
        except ImportError:
            print("[WARN] face_recognition_style_detection module not found, skipping face recognition matching")
        except Exception as e:
            print(f"[WARN] Face recognition matching error: {e}")
    
    # 3. SECONDARY MODEL: Compare with yolo_files database (if face recognition didn't match)
    # This uses yolo_files table as SECONDARY model
    comparison_result = compare_with_yolo_files_database(image_path)
    
    # If we found a match in yolo_files database, use it
    if comparison_result.get("matched") and comparison_result.get("confidence", 0) > 0.5:
        matched_condition = comparison_result.get("condition", "Healthy")
        matched_confidence = comparison_result.get("confidence", 0.0)
        matched_disease = comparison_result.get("disease_name")
        matched_treatment = comparison_result.get("treatment")
        
        print(f"[SECONDARY MODEL] Matched with yolo_files database: {matched_condition} (confidence: {matched_confidence:.1%})")
        if matched_disease:
            print(f"   Disease: {matched_disease}")
        if matched_treatment:
            print(f"   Treatment available: {matched_treatment[:50]}...")
        
        image_features = analyze_image_features(image_path)
        return {
            "condition": matched_condition,
            "disease_name": matched_disease,
            "treatment": matched_treatment,
            "confidence": matched_confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
            "all_probabilities": {matched_condition: matched_confidence},
            "recommendations": get_recommendations(matched_condition, matched_confidence, {matched_condition: matched_confidence}, image_features, matched_disease, matched_treatment),
            "matched_from_database": True,
            "matched_file": comparison_result.get("matched_file"),
            "detection_method": "yolo_files_database_secondary"
        }
    
    # 4. If no database match (face recognition or yolo_files), log that we're using AI as support/fallback
    if not comparison_result.get("matched") and not (face_recognition_match and face_recognition_match.get("matched")):
        print("[SUPPORT] No match in petchay_dataset or yolo_files, using AI models (CNN/YOLO) as support...")
    
    # 2.5. Check exact filename match in yolo_files
    existing_yolo_file = None
    try:
        from db import get_yolo_file_by_filename
        existing_yolo_file = get_yolo_file_by_filename(filename)
    except Exception as e:
        print(f"Error checking yolo_files: {e}")

    if not existing_yolo_file:
        print(f"No match found in yolo_files for {filename}. Trying YOLO-only detection.")

    if existing_yolo_file:
        print(f"Match found in yolo_files for {filename}. Proceeding with AI enhancement.")

    # Prediction logic
    if not yolo_model and not cnn_predictor:
        try:
            load_model_background()
        except Exception as e:
            print(f"Error loading models on-demand: {e}")

    if yolo_model:
        result = yolo_predict_image(image_path, conf_threshold=0.25 if existing_yolo_file else 0.35)
        if result.get("success"):
            condition = result.get("condition", "Healthy")
            confidence = result.get("confidence", 0.0)
            probabilities = result.get("all_probabilities", {})
            image_features = analyze_image_features(image_path)

            # Check Roboflow for Blackrot if likely a Pechay
            if condition != "Not Pechay":
                roboflow_res = check_roboflow_disease(image_path)
                if roboflow_res:
                    condition = "Diseased"
                    confidence = roboflow_res["confidence"]
                    probabilities = {"Blackrot": confidence}

            if condition == "Not Pechay":
                # Trust YOLO's judgement on object detection.
                # If YOLO doesn't see a Pechay leaf, we shouldn't force a classification.
                return {
                    "condition": "Not Pechay",
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
                    "all_probabilities": {},
                    "recommendations": {
                        "title": "Not Pechay",
                        "tips": ["The system could not identify this as a Pechay leaf.", "Please upload a clear Pechay leaf image."],
                        "action": "Retry Upload",
                        "urgency": "low"
                    },
                    "success": False
                }
            # Get treatment from existing_yolo_file if available
            treatment = None
            disease_name = None
            if existing_yolo_file:
                treatment = existing_yolo_file.get("treatment")
                dataset_type = existing_yolo_file.get("dataset_type", "")
                if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                    disease_name = dataset_type
            
            # Get disease_name and treatment from existing_yolo_file if available
            if not disease_name or not treatment:
                if existing_yolo_file:
                    if not disease_name:
                        dataset_type = existing_yolo_file.get("dataset_type", "")
                        if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                            disease_name = dataset_type
                    if not treatment:
                        treatment = existing_yolo_file.get("treatment")
            
            return {
                "condition": condition,
                "disease_name": disease_name,
                "treatment": treatment,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
                "all_probabilities": probabilities,
                "recommendations": get_recommendations(condition, confidence, probabilities, image_features, disease_name, treatment),
            }
        else:
            print(f"YOLO prediction failed: {result.get('error', 'Unknown error')}")

    if cnn_predictor:
        
        # Use trained CNN model (simulating YOLOv9 enhancement)
        result = cnn_predictor.predict_image(image_path)
        
        if result['success']:
            condition = result['condition']
            confidence = result['confidence']
            probabilities = result.get('all_probabilities', {})
            if HEALTHY_ONLY_MODE and condition == "Diseased":
                condition = "Healthy"
                probabilities = {"Healthy": confidence}
            
            # Check Roboflow for Blackrot (overrides Healthy Only if found)
            disease_name = None
            # Roboflow API Disabled per user request to "dont use api"
            # if condition != "Not Pechay":
            #      roboflow_res = check_roboflow_disease(image_path)
            #      if roboflow_res:
            #          condition = "Diseased"
            #          confidence = roboflow_res["confidence"]
            #          probabilities = {"Blackrot": confidence}
            #          disease_name = roboflow_res.get("disease_name", "Blackrot")
            
            # If condition is "Not Pechay" (from the model itself), handle it
            if condition == "Not Pechay":
                 return {
                    "condition": "Not Pechay",
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
                    "all_probabilities": probabilities,
                    "recommendations": result.get('recommendations', {
                        "title": "⚠️ Object Not Recognized",
                        "tips": ["Please upload a clear image of a Pechay leaf", "Ensure good lighting and focus", "Remove background clutter"],
                        "action": "Upload a valid Pechay leaf image",
                        "urgency": "low"
                    })
                }

            # Analyze image features for specific recommendations
            image_features = analyze_image_features(image_path)
            
            # Get treatment from existing_yolo_file if available
            treatment = None
            if existing_yolo_file:
                treatment = existing_yolo_file.get("treatment")
                # Also get disease_name from dataset_type if not already set
                if not disease_name:
                    dataset_type = existing_yolo_file.get("dataset_type", "")
                    if dataset_type and dataset_type not in ["Healthy", "Diseased"]:
                        disease_name = dataset_type
            
            return {
                "condition": condition,
                "disease_name": disease_name,
                "treatment": treatment,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
                "all_probabilities": probabilities,
                "recommendations": get_recommendations(condition, confidence, probabilities, image_features, disease_name, treatment),
                "detection_method": "cnn_ai_support"  # AI used as support
            }
        else:
            print(f"CNN prediction failed: {result.get('error', 'Unknown error')}")
            # Fallback if AI fails even if match found
            return {
                "condition": "Error",
                "confidence": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": image_path.replace(BASE_DIR + os.sep, "").replace("\\", "/"),
                "recommendations": {
                    "title": "Prediction Error",
                    "tips": ["An error occurred during AI analysis."],
                    "action": "Try again",
                    "urgency": "low"
                }
            }
    
    # Final fallback: Use hybrid detection (only if database and AI support both failed)
    print("[FALLBACK] Using hybrid detection as last resort...")
    result = hybrid_detection(image_path)
    result["detection_method"] = "hybrid_fallback"
    return result


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Lightweight prediction API for devices (e.g., ESP32-CAM). No login required."""
    # Check for 'file' (ESP32 standard) or 'leafImage' (legacy/web)
    if 'file' in request.files:
        file = request.files['file']
    elif 'leafImage' in request.files:
        file = request.files['leafImage']
    else:
        return jsonify({'error': 'No file part'}), 400

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        # Ensure unique filename if empty or generic
        if not filename or filename == "image.jpg":
             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
             filename = f"esp32_{timestamp}.jpg"
        else:
             # Add UUID to prevent overwrites
             filename = f"{uuid.uuid4().hex[:8]}_{filename}"
             
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Run detection
        # Note: detect_leaf_condition returns a dict, not a tuple
        result = detect_leaf_condition(filepath)
        label = result.get("condition", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        # Save to database
        try:
            # Ensure recommendations exist
            recommendations = result.get("recommendations")
            if not recommendations:
                image_features = analyze_image_features(filepath)
                recommendations = get_recommendations(
                    label, 
                    confidence, 
                    result.get("all_probabilities"), 
                    image_features,
                    result.get("disease_name"),
                    result.get("treatment")
                )

            create_detection_result(
                filename=filename,
                condition=label,
                confidence=confidence,
                image_path=os.path.join("uploads", filename).replace("\\", "/"),
                recommendations=recommendations,
                all_probabilities=result.get("all_probabilities"),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                disease_name=result.get("disease_name"),
                treatment=result.get("treatment")
            )
        except Exception as e:
            print(f"Error saving to database: {e}")

        return jsonify({
            'label': label,
            'confidence': float(confidence)
        })


@app.route("/dataset_manager")
def dataset_manager():
    gate = require_login()
    if gate is not None:
        return gate
    
    # Get stats/list of custom data
    raw = list(reversed(get_custom_training_data()))  # Show newest first
    dataset_images = []
    for entry in raw:
        if isinstance(entry, dict):
            label = entry.get("disease_name") or entry.get("label") or entry.get("condition") or "Unlabeled"
            dataset_images.append({**entry, "label": label})
    
    return render_template("dataset_manager.html", dataset_images=dataset_images)

@app.route("/upload_training_data", methods=["POST"])
def upload_training_data():
    gate = require_login()
    if gate is not None:
        return gate

    username = session.get("user")
    current_user_id = None
    if username and username != "admin":
        try:
            user = get_user_by_username(username)
            if user:
                current_user_id = user.get("id")
        except Exception:
            current_user_id = None
        
    label = request.form.get("label")
    custom_label = request.form.get("custom_label")
    if label == "Other" and custom_label:
        label = custom_label
        
    files = request.files.getlist("images")
    
    saved_count = 0
    
    if cnn_predictor is None:
        try:
            load_model_background()
        except Exception:
            pass

    condition = "Healthy" if (label or "").strip() == "Healthy" else "Diseased"
    disease_name = None if condition == "Healthy" else (label or "").strip() or None
    folder_label = "Healthy" if condition == "Healthy" else (disease_name or "Diseased")

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create unique filename
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            
            # Save to specific dataset folder
            dataset_dir = os.path.join(UPLOAD_FOLDER, "dataset_custom", folder_label)
            os.makedirs(dataset_dir, exist_ok=True)
            
            save_path = os.path.join(dataset_dir, unique_filename)
            file.save(save_path)
            
            # Validate if image is actually a pechay leaf
            gate = pechay_color_gate(save_path)
            if not gate.get("ok", True):
                # Delete invalid image
                try:
                    os.remove(save_path)
                except:
                    pass
                continue  # Skip this file and move to next
            
            # Extract features immediately
            embedding = None
            if cnn_predictor:
                try:
                    features = cnn_predictor.extract_features(save_path)
                    if features is not None:
                        embedding = features.tolist()
                except Exception as e:
                    print(f"Error extracting features for {filename}: {e}")
            
            # Upload to Supabase Storage for persistent URL
            dataset_bucket = os.getenv("DATASET_BUCKET_NAME", "petchay-images")  # Use petchay-images if not set
            print(f"[UPLOAD] Uploading to bucket: {dataset_bucket}")
            image_url = upload_image_to_storage(save_path, bucket_name=dataset_bucket)
            if image_url:
                print(f"[UPLOAD] ✅ Image uploaded to storage: {image_url}")
            else:
                print(f"[UPLOAD] ⚠️ Storage upload failed, using local path")
                # Fallback to local path if upload fails
                image_url = f"/uploads/dataset_custom/{folder_label}/{unique_filename}"

            # Save metadata to DB (Supabase + Local JSON)
            save_dataset_entry(
                filename=unique_filename,
                label=condition,
                image_url=image_url,
                embedding=embedding,
                disease_name=disease_name,
                user_id=current_user_id
            )
            saved_count += 1
            
    # Reload embeddings in memory
    load_embeddings()
    
    flash(f"Successfully added {saved_count} images to the '{folder_label}' dataset!", "success")
    return redirect(url_for("dataset_manager"))


@app.route("/upload_image_immediate", methods=["POST"])
def upload_image_immediate():
    """AJAX endpoint to save images immediately when selected"""
    try:
        # Check authentication without redirect
        if not session.get("user"):
            return jsonify({"success": False, "error": "Not authenticated"}), 401

        username = session.get("user")
        current_user_id = None
        if username and username != "admin":
            try:
                user = get_user_by_username(username)
                if user:
                    current_user_id = user.get("id")
            except Exception:
                current_user_id = None
        
        # Get form data
        condition = request.form.get("condition", "Healthy")
        disease_select = request.form.get("disease_select", "")
        new_disease_name = request.form.get("new_disease_name", "").strip()
        treatment = request.form.get("treatment", "").strip()
        
        # Determine final disease name
        disease_name = ""
        if condition == "Diseased":
            if disease_select == "new":
                disease_name = new_disease_name if new_disease_name else "Unknown Disease"
            else:
                disease_name = disease_select
        
        # Folder label for directory structure
        folder_label = "Healthy" if condition == "Healthy" else (disease_name or "Diseased")
        
        # Handle single file
        file = request.files.get("file")
        if not file:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        if not file.filename or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type. Only images are allowed."}), 400
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        
        # Save to specific dataset folder
        dataset_dir = os.path.join(UPLOAD_FOLDER, "dataset_custom", folder_label)
        os.makedirs(dataset_dir, exist_ok=True)
        
        save_path = os.path.join(dataset_dir, unique_filename)
        file.save(save_path)
        
        # Validate if image is actually a pechay leaf
        gate = pechay_color_gate(save_path)
        if not gate.get("ok", True):
            # Delete invalid image
            try:
                os.remove(save_path)
            except:
                pass
            
            reason = gate.get("reason", "unknown")
            reason_messages = {
                "digital_image": "Image appears to be digital/drawn. Please upload a real photo of a pechay leaf.",
                "round_shape": "Image appears to be round. Pechay leaves are not round.",
                "not_green_enough": "Image does not contain enough green color typical of pechay leaves.",
                "not_green_dominant": "Green color is not dominant. This may not be a pechay leaf.",
                "green_color_out_of_range": "The green color is outside the typical pechay leaf color range.",
                "person_like": "Image appears to contain skin tones, not a pechay leaf.",
                "empty": "Image file is empty or corrupted."
            }
            error_msg = reason_messages.get(reason, f"Image validation failed: {reason}")
            return jsonify({"success": False, "error": error_msg, "reason": reason}), 400
        
        # Extract features immediately
        embedding = None
        if cnn_predictor:
            try:
                features = cnn_predictor.extract_features(save_path)
                if features is not None:
                    embedding = features.tolist()
            except Exception as e:
                print(f"Error extracting features for {filename}: {e}")
        
        # Upload to Supabase Storage
        dataset_bucket = os.getenv("DATASET_BUCKET_NAME", "petchay-images")  # Use petchay-images if not set
        print(f"[UPLOAD] Uploading to bucket: {dataset_bucket}")
        image_url = upload_image_to_storage(save_path, bucket_name=dataset_bucket)
        if image_url:
            print(f"[UPLOAD] ✅ Image uploaded to storage: {image_url}")
        else:
            print(f"[UPLOAD] ⚠️ Storage upload failed, using local path")
            image_url = f"/uploads/dataset_custom/{folder_label}/{unique_filename}"
        
        # Save to yolo_files table for scanning
        try:
            create_yolo_file(
                filename=unique_filename,
                file_type="image",
                dataset_type=folder_label,
                url=image_url,
                treatment=treatment if condition == "Diseased" and treatment else None
            )
        except Exception as e:
            print(f"Error saving to yolo_files: {e}")
        
        # Save metadata to dataset
        save_dataset_entry(
            filename=unique_filename,
            label=condition,
            image_url=image_url,
            embedding=embedding,
            disease_name=(disease_name if condition == "Diseased" else None),
            user_id=current_user_id
        )
        
        return jsonify({
            "success": True,
            "filename": unique_filename,
            "message": f"Image '{filename}' saved successfully!"
        })
    except Exception as e:
        print(f"Error in upload_image_immediate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload_dataset_workflow", methods=["POST"])
def upload_dataset_workflow():
    """Final endpoint to complete upload and trigger training"""
    gate = require_login()
    if gate is not None:
        return gate

    username = session.get("user")
    current_user_id = None
    if username and username != "admin":
        try:
            user = get_user_by_username(username)
            if user:
                current_user_id = user.get("id")
        except Exception:
            current_user_id = None
    
    # Get form data
    condition = request.form.get("condition", "Healthy")
    disease_select = request.form.get("disease_select", "")
    new_disease_name = request.form.get("new_disease_name", "").strip()
    treatment = request.form.get("treatment", "").strip()
    
    # Determine final disease name
    disease_name = ""
    if condition == "Diseased":
        if disease_select == "new":
            disease_name = new_disease_name if new_disease_name else "Unknown Disease"
        else:
            disease_name = disease_select
    
    # Folder label for directory structure
    folder_label = "Healthy" if condition == "Healthy" else (disease_name or "Diseased")
    
    # Handle any remaining files (if user uploaded more after initial batch)
    files = request.files.getlist("files")
    saved_count = 0
    
    # Ensure model is loaded for feature extraction
    if cnn_predictor is None:
        try:
            load_model_background()
        except Exception:
            pass
            
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            
            # Save to specific dataset folder: uploads/dataset_custom/{folder_label}
            dataset_dir = os.path.join(UPLOAD_FOLDER, "dataset_custom", folder_label)
            os.makedirs(dataset_dir, exist_ok=True)
            
            save_path = os.path.join(dataset_dir, unique_filename)
            file.save(save_path)
            
            # Extract features immediately
            embedding = None
            if cnn_predictor:
                try:
                    features = cnn_predictor.extract_features(save_path)
                    if features is not None:
                        embedding = features.tolist()
                except Exception as e:
                    print(f"Error extracting features for {filename}: {e}")
            
            # Upload to Supabase Storage (optional but good for persistence)
            dataset_bucket = os.getenv("DATASET_BUCKET_NAME", "petchay-images")  # Use petchay-images if not set
            print(f"[UPLOAD] Uploading to bucket: {dataset_bucket}")
            image_url = upload_image_to_storage(save_path, bucket_name=dataset_bucket)
            if image_url:
                print(f"[UPLOAD] ✅ Image uploaded to storage: {image_url}")
            else:
                print(f"[UPLOAD] ⚠️ Storage upload failed, using local path")
            if not image_url:
                # Local fallback URL
                image_url = f"/uploads/dataset_custom/{folder_label}/{unique_filename}"
            
            # Save to yolo_files
            try:
                create_yolo_file(
                    filename=unique_filename,
                    file_type="image",
                    dataset_type=folder_label,
                    url=image_url,
                    treatment=treatment if condition == "Diseased" and treatment else None
                )
            except Exception as e:
                print(f"Error saving to yolo_files: {e}")
            
            # Save metadata
            save_dataset_entry(
                filename=unique_filename,
                label=condition,
                image_url=image_url,
                embedding=embedding,
                disease_name=(disease_name if condition == "Diseased" else None),
                user_id=current_user_id
            )
            saved_count += 1
            
    # Reload embeddings in memory to update the model immediately
    load_embeddings()
    
    # Trigger background training (YOLO)
    threading.Thread(target=trigger_training_and_reload, daemon=True).start()
    
    flash(f"Upload complete! {saved_count} additional images saved. YOLO training started in background.", "success")
    return redirect(url_for("dataset_manager"))


@app.route("/upload_smart", methods=["POST"])
def upload_smart():
    gate = require_login()
    if gate is not None:
        return gate

    username = session.get("user")
    current_user_id = None
    if username and username != "admin":
        try:
            user = get_user_by_username(username)
            if user:
                current_user_id = user.get("id")
        except Exception:
            current_user_id = None
    
    label = (request.form.get("label") or "").strip()
    condition = (request.form.get("condition") or "").strip()
    disease_name = (request.form.get("disease_name") or "").strip()

    if condition not in ("Healthy", "Diseased"):
        if label.lower() == "healthy":
            condition = "Healthy"
        elif label:
            condition = "Diseased"
            if not disease_name:
                disease_name = label
        else:
            condition = "Diseased"

    if condition == "Healthy":
        disease_name = ""

    folder_label = "Healthy" if condition == "Healthy" else (disease_name or "Diseased")
        
    files = request.files.getlist("files")
    
    saved_count = 0

    if cnn_predictor is None:
        try:
            load_model_background()
        except Exception:
            pass
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            
            # Save to specific dataset folder
            dataset_dir = os.path.join(UPLOAD_FOLDER, "dataset_custom", folder_label)
            os.makedirs(dataset_dir, exist_ok=True)
            
            save_path = os.path.join(dataset_dir, unique_filename)
            file.save(save_path)
            
            # Extract features immediately (for "Review Model then emedings" part)
            embedding = None
            if cnn_predictor:
                try:
                    features = cnn_predictor.extract_features(save_path)
                    if features is not None:
                        embedding = features.tolist()
                except Exception as e:
                    print(f"Error extracting features for {filename}: {e}")
            
            # Upload to Supabase Storage
            dataset_bucket = os.getenv("DATASET_BUCKET_NAME", "petchay-images")  # Use petchay-images if not set
            print(f"[UPLOAD] Uploading to bucket: {dataset_bucket}")
            image_url = upload_image_to_storage(save_path, bucket_name=dataset_bucket)
            if image_url:
                print(f"[UPLOAD] ✅ Image uploaded to storage: {image_url}")
            else:
                print(f"[UPLOAD] ⚠️ Storage upload failed, using local path")
            if not image_url:
                image_url = f"/uploads/dataset_custom/{folder_label}/{unique_filename}"

            # Save metadata
            save_dataset_entry(
                filename=unique_filename,
                label=condition,
                image_url=image_url,
                embedding=embedding,
                disease_name=(disease_name or None),
                user_id=current_user_id
            )
            saved_count += 1
            
    # Reload embeddings
    load_embeddings()
    
    flash(f"Successfully processed {saved_count} items for '{folder_label}'!", "success")
    return redirect(url_for("dataset_manager"))



def require_login():
    if not session.get("user"):
        return redirect(url_for("login"))
    return None


def load_users():
    """Load users from Supabase (for backward compatibility)"""
    try:
        from db import get_all_users
        return get_all_users()
    except Exception as e:
        print(f"Error loading users from database: {e}")
        return []

def save_users(users):
    """Save users to Supabase (for backward compatibility) - Not used anymore"""
    pass  # Users are now saved directly via create_user

@app.route("/login", methods=["GET", "POST", "HEAD"]) 
@app.route("/", methods=["GET", "POST", "HEAD"]) 
def login():
    error = None
    success = None
    mode = request.args.get('mode', 'login')
    
    if request.method == "POST":
        if request.form.get('action') == 'register':
            # Registration handling
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            
            # Validation
            if not username or not email or not password or not confirm_password:
                error = "All fields are required!"
            elif len(username) < 3:
                error = "Username must be at least 3 characters!"
            elif len(password) < 4:
                error = "Password must be at least 4 characters!"
            elif password != confirm_password:
                error = "Passwords do not match!"
            else:
                try:
                    # Check if username already exists
                    if get_user_by_username(username):
                        error = "Username already exists!"
                    # Check if email already exists
                    elif get_user_by_email(email):
                        error = "Email already registered!"
                    else:
                        # Create new user in Supabase
                        password_hash = generate_password_hash(password)
                        create_user(username, email, password_hash)
                        success = "Registration successful! You can now login."
                        mode = 'login'
                except DatabaseError as e:
                    error = str(e)
                except Exception as e:
                    error = f"Registration failed: {str(e)}"
        else:
            # Login handling
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            
            # Check default admin account
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123") # Default if not set, but should be set in .env
            if username == "admin" and password == admin_password:
                session["user"] = username
                return redirect(url_for("dashboard"))
            
            # Check registered users in Supabase
            try:
                user = get_user_by_username(username)
                if user and check_password_hash(user.get('password', ''), password):
                    session["user"] = username
                    return redirect(url_for("dashboard"))
                else:
                    error = "Invalid username or password!"
            except DatabaseError as e:
                error = f"Login error: {str(e)}"
            except Exception as e:
                error = f"Login failed: {str(e)}"
    
    return render_template("login.html", error=error, success=success, mode=mode)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"]) 
def dashboard():
    gate = require_login()
    if gate is not None:
        return gate

    page = request.args.get("page", "dashboard")
    upload_status = ""
    detection_result = None
    settings_message = None
    current_user = None
    
    # Get current user information for settings page
    if session.get("user"):
        try:
            current_user = get_user_by_username(session["user"])
            if current_user:
                # Don't pass password to template
                current_user.pop("password", None)
        except Exception as e:
            print(f"Error fetching current user: {e}")
    
    # Get current user ID for filtering results
    current_user_id = None
    is_admin = False
    if session.get("user"):
        username = session.get("user")
        # Check if admin account
        if username == "admin":
            is_admin = True
            # Admin can see all results, so don't filter by user_id
            current_user_id = None
        else:
            try:
                user = get_user_by_username(username)
                if user:
                    current_user_id = user.get("id")
            except Exception as e:
                print(f"Error fetching current user: {e}")
    
    # Calculate dashboard statistics from Supabase
    # Admin sees all results, regular users see only their own
    try:
        dashboard_stats = get_dashboard_stats(user_id=None if is_admin else current_user_id)
        # In healthy only mode, we ensure diseased count is 0 for display purposes
        if HEALTHY_ONLY_MODE:
             dashboard_stats['diseased_leaves'] = 0
             if dashboard_stats['total_scans'] > 0:
                 dashboard_stats['success_rate'] = int((dashboard_stats['healthy_leaves'] / dashboard_stats['total_scans']) * 100)
    except Exception as e:
        print(f"Error fetching dashboard stats: {e}")
        dashboard_stats = {
            'total_scans': 0,
            'healthy_leaves': 0,
            'diseased_leaves': 0,
            'success_rate': 0
        }

    recent_scans = []
    try:
        db_recent_scans = get_all_detection_results(limit=6, user_id=None if is_admin else current_user_id)
        for scan in db_recent_scans:
            ts = scan.get("timestamp")
            scan_dt = None
            if isinstance(ts, str) and ts:
                try:
                    if "T" in ts:
                        scan_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        scan_dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    scan_dt = datetime.now()
            else:
                scan_dt = datetime.now()

            image_path = scan.get("image_path", "")
            if isinstance(image_path, str):
                if image_path.startswith("/"):
                    image_path = image_path[1:]
                if not image_path.startswith("uploads/") and image_path:
                    image_path = f"uploads/{image_path.lstrip('/')}"

            recent_scans.append(
                {
                    "condition": scan.get("condition", "Healthy"),
                    "timestamp": scan_dt,
                    "image_path": image_path,
                }
            )
    except Exception as e:
        print(f"Error fetching recent scans: {e}")

    # Dataset page logic
    dataset_stats = None
    dataset_images = []
    if page == "dataset":
        try:
            dataset_stats = get_dataset_stats()
            # Get filter
            dataset_label = request.args.get("label")
            dataset_images = get_dataset_images(limit=100, label=dataset_label)
        except Exception as e:
            print(f"Error fetching dataset info: {e}")
            dataset_stats = {"total_images": 0, "healthy_count": 0, "diseased_count": 0}

    # Handle dataset upload
    if request.method == "POST" and page == "dataset" and is_admin:
        action = request.form.get("action")
        if action == "upload_dataset":
            files = request.files.getlist("dataset_files")
            label = request.form.get("dataset_label")
            split = request.form.get("dataset_split", "train")
            author = request.form.get("dataset_author", "")
            license = request.form.get("dataset_license", "")
            disease_type = request.form.get("disease_type", "")
            
            if not files or not label:
                upload_status = "Please select files and a label."
            else:
                count = 0
                for file in files:
                    if file and allowed_file(file.filename):
                        # Save file
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        unique_id = str(uuid.uuid4())[:8]
                        # Sanitize filename
                        safe_name = secure_filename(file.filename)
                        filename = f"{label}_{timestamp}_{unique_id}_{safe_name}"
                        
                        # Save to dataset folder
                        save_path = os.path.join(UPLOAD_FOLDER, "dataset", label)
                        os.makedirs(save_path, exist_ok=True)
                        full_path = os.path.join(save_path, filename)
                        file.save(full_path)
                        
                        # Log to file_uploads table (new separate table)
                        try:
                            file_size = os.path.getsize(full_path)
                            create_file_upload_log(
                                filename=filename,
                                file_path=full_path,
                                original_name=file.filename,
                                file_size=file_size,
                                upload_source="dataset",
                                user_id=session.get("user_id")
                            )
                        except Exception as e:
                            print(f"Error logging to file_uploads: {e}")

                        # Add to DB
                        try:
                            metadata = {
                                "author": author,
                                "license": license
                            }
                            if label == "Diseased" and disease_type:
                                metadata["disease_type"] = disease_type
                                
                            create_dataset_image(
                                filename=filename,
                                label=label,
                                image_path=os.path.join("uploads", "dataset", label, filename).replace("\\", "/"),
                                split=split,
                                metadata=metadata,
                                user_id=current_user_id
                            )
                            
                            # Also add to yolo_files to make it a "known" file for detection
                            try:
                                create_yolo_file(
                                    filename=filename,
                                    file_type="image",
                                    dataset_type=label, # Use label (Healthy/Diseased) as dataset_type
                                    url=os.path.join("uploads", "dataset", label, filename).replace("\\", "/")
                                )
                            except Exception as e:
                                print(f"Error adding to yolo_files: {e}")
                                
                            count += 1
                        except Exception as e:
                            print(f"Error adding to dataset: {e}")
            
                upload_status = f"Successfully added {count} images to {label} dataset ({split})."
                # Refresh data
                try:
                    dataset_stats = get_dataset_stats()
                    dataset_images = get_dataset_images(limit=100)
                except:
                    pass

    # Handle settings page actions
    if request.method == "POST" and page == "settings":
        action = request.form.get("action")
        username = session.get("user")
        
        if not username:
            settings_message = "Error: Not logged in"
        elif action == "update_profile":
            try:
                new_email = request.form.get("email", "").strip()
                if not new_email:
                    settings_message = "Error: Email is required"
                else:
                    update_user_email(username, new_email)
                    settings_message = "✓ Profile updated successfully!"
                    # Refresh user data
                    current_user = get_user_by_username(username)
                    if current_user:
                        current_user.pop("password", None)
            except DatabaseError as e:
                settings_message = f"Error: {str(e)}"
            except Exception as e:
                settings_message = f"Error updating profile: {str(e)}"
        
        elif action == "change_password":
            try:
                current_password = request.form.get("current_password", "")
                new_password = request.form.get("new_password", "")
                confirm_password = request.form.get("confirm_password", "")
                
                if not current_password or not new_password or not confirm_password:
                    settings_message = "Error: All password fields are required"
                elif new_password != confirm_password:
                    settings_message = "Error: New passwords do not match"
                elif len(new_password) < 4:
                    settings_message = "Error: New password must be at least 4 characters"
                else:
                    # Verify current password
                    user = get_user_by_username(username)
                    if not user:
                        settings_message = "Error: User not found"
                    elif not check_password_hash(user.get("password", ""), current_password):
                        settings_message = "Error: Current password is incorrect"
                    else:
                        # Update password
                        new_password_hash = generate_password_hash(new_password)
                        update_user_password(username, new_password_hash)
                        settings_message = "✓ Password changed successfully!"
            except DatabaseError as e:
                settings_message = f"Error: {str(e)}"
            except Exception as e:
                settings_message = f"Error changing password: {str(e)}"
        
        elif action == "delete_account":
            try:
                # Delete user account
                delete_user(username)
                settings_message = "Account deleted successfully. Redirecting to login..."
                session.pop("user", None)
                # Redirect after a short delay
                return redirect(url_for("login"))
            except DatabaseError as e:
                settings_message = f"Error: {str(e)}"
            except Exception as e:
                settings_message = f"Error deleting account: {str(e)}"

    if request.method == "POST" and page == "upload":
        file = request.files.get("leafImage")
        if not file or file.filename == "":
            upload_status = "File is required."
        elif not allowed_file(file.filename):
            upload_status = "File is not an image."
        else:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            upload_status = "File uploaded successfully."
            detection_result = detect_leaf_condition(save_path)
            
            # Store detection result in Supabase
            try:
                user_id = None
                username = session.get("user")
                if username:
                    # For admin, we can leave user_id as None or try to find/create admin user
                    if username != "admin":
                        user = get_user_by_username(username)
                        if user:
                            user_id = user.get("id")
                    # Admin uploads will have user_id=None, which is fine
                    # Admin can see all results including those with NULL user_id
                
                create_detection_result(
                    filename=filename,
                    condition=detection_result["condition"],
                    confidence=detection_result.get("confidence", 0),
                    image_path=detection_result["image_path"],
                    recommendations=detection_result.get("recommendations", {}),
                    all_probabilities=detection_result.get("all_probabilities"),
                    user_id=user_id,
                    timestamp=detection_result.get("timestamp"),
                    disease_name=detection_result.get("disease_name"),
                    treatment=detection_result.get("treatment")
                )
            except Exception as e:
                print(f"Error saving detection result to database: {e}")
                upload_status += " (Note: Result saved but database storage failed)"

    # Build results from Supabase (filtered by current user, except admin sees all)
    results = []
    filter_condition = request.args.get("filter", "").strip()  # Get filter parameter (Healthy/Diseased)
    filter_days = request.args.get("days", "").strip()  # Get date filter parameter (1, 7, 21, or empty for all)
    
    # Convert days filter to integer and calculate date range
    days_filter = None
    date_range_text = None
    if filter_days:
        try:
            days_filter = int(filter_days)
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_filter)
            # Format dates nicely
            date_range_text = f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"
        except ValueError:
            days_filter = None
    
    if page == "results":
        try:
            # Admin sees all results, regular users see only their own
            filter_user_id = None if is_admin else current_user_id
            if filter_condition:
                db_results = get_detection_results_by_condition(filter_condition, user_id=filter_user_id, days=days_filter)
            else:
                db_results = get_all_detection_results(limit=1000, user_id=filter_user_id, days=days_filter)
            
            for db_result in db_results:
                # Convert database result to template format
                timestamp_str = db_result.get("timestamp", "")
                if isinstance(timestamp_str, str):
                    try:
                        # Try parsing ISO format or standard format
                        if 'T' in timestamp_str:
                            result_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            result_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except:
                        result_timestamp = datetime.now()
                else:
                    result_timestamp = datetime.now()
                
                filename = db_result.get("filename", "")
                # Check if file exists locally
                fpath = os.path.join(UPLOAD_FOLDER, filename)
                
                # Get image path from database or construct it
                image_path = db_result.get("image_path", "")
                if not image_path:
                    # Fallback: construct path from filename
                    image_path = f"uploads/{filename}"
                # Ensure path uses forward slashes and doesn't start with /
                if image_path.startswith("/"):
                    image_path = image_path[1:]
                if image_path.startswith("uploads/"):
                    pass  # Already correct
                elif not image_path.startswith("uploads"):
                    image_path = f"uploads/{image_path}"
                
                # Clean filename - ensure it's just the basename (no path)
                clean_filename = os.path.basename(filename) if filename else ""
                # Update fpath to use clean filename
                fpath = os.path.join(UPLOAD_FOLDER, clean_filename)
                file_exists = os.path.exists(fpath) if clean_filename else False
                
                result_data = {
                    "filename": clean_filename,  # Use clean filename for URL
                    "path": f"uploads/{clean_filename}" if clean_filename else "",
                    "timestamp": result_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_obj": result_timestamp,  # For sorting
                    "condition": db_result.get("condition", "Healthy"),
                    "disease_name": db_result.get("disease_name"),  # Include disease name if available
                    "confidence": float(db_result.get("confidence", 0)),
                    "recommendations": db_result.get("recommendations", {}),
                    "file_exists": file_exists
                }
                
                # If recommendations don't exist or are empty, generate them for display
                if not result_data["recommendations"] or not isinstance(result_data["recommendations"], dict):
                    if os.path.exists(fpath):
                        image_features = analyze_image_features(fpath)
                        result_data["recommendations"] = get_recommendations(
                            result_data["condition"],
                            result_data["confidence"],
                            db_result.get("all_probabilities"),
                            image_features
                        )
                        # Optionally update in database (commented out to avoid unnecessary writes)
                        # try:
                        #     update_detection_result(
                        #         filename=filename,
                        #         recommendations=result_data["recommendations"]
                        #     )
                        # except:
                        #     pass
                
                results.append(result_data)
            
            # Sort results by timestamp (newest first)
            results.sort(key=lambda x: x["timestamp_obj"], reverse=True)
            
            # Remove timestamp_obj before passing to template
            for result in results:
                result.pop("timestamp_obj", None)
                
        except Exception as e:
            print(f"Error fetching results from database: {e}")
            results = []

    # Get database URL for display (masked for security)
    db_url_display = None
    try:
        from dotenv import load_dotenv
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL", "")
        if supabase_url:
            # Show only domain part for display
            if "supabase.co" in supabase_url:
                db_url_display = supabase_url.split("//")[-1] if "//" in supabase_url else supabase_url
            else:
                db_url_display = "Configured"
    except:
        db_url_display = "Not configured"
    
    # Analytics Data
    analytics_data = None
    if page == "analytics":
        try:
            # Fetch all results for analytics (respecting user permissions)
            # Use the same filter_user_id logic as above
            filter_user_id = None if is_admin else current_user_id
            analytics_results = get_all_detection_results(limit=10000, user_id=filter_user_id)
            
            # 1. Daily Scans (Last 7 days)
            daily_scans = {}
            from datetime import timedelta
            today = datetime.now().date()
            for i in range(6, -1, -1):
                day = today - timedelta(days=i)
                daily_scans[day.strftime('%Y-%m-%d')] = 0
                
            for res in analytics_results:
                ts = res.get("timestamp")
                if ts:
                    try:
                        # Handle ISO format
                        if 'T' in ts:
                            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else:
                             dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        date_str = dt.date().strftime('%Y-%m-%d')
                        if date_str in daily_scans:
                            daily_scans[date_str] += 1
                    except:
                        pass
            
            # 2. Confidence Distribution
            confidence_dist = {"0-50%": 0, "50-70%": 0, "70-90%": 0, "90-100%": 0}
            for res in analytics_results:
                try:
                    conf = float(res.get("confidence", 0))
                    if conf < 50:
                        confidence_dist["0-50%"] += 1
                    elif conf < 70:
                        confidence_dist["50-70%"] += 1
                    elif conf < 90:
                        confidence_dist["70-90%"] += 1
                    else:
                        confidence_dist["90-100%"] += 1
                except:
                    pass
                    
            analytics_data = {
                "daily_labels": list(daily_scans.keys()),
                "daily_values": list(daily_scans.values()),
                "confidence_labels": list(confidence_dist.keys()),
                "confidence_values": list(confidence_dist.values()),
                "total_scans": len(analytics_results),
                "healthy_count": sum(1 for r in analytics_results if r.get("condition") == "Healthy"),
                "diseased_count": sum(1 for r in analytics_results if r.get("condition") == "Diseased")
            }
        except Exception as e:
            print(f"Error generating analytics: {e}")

    return render_template(
        "dashboard.html",
        page=page,
        upload_status=upload_status,
        detection_result=detection_result,
        recent_scans=recent_scans,
        results=results,
        dashboard_stats=dashboard_stats,
        analytics_data=analytics_data,
        filter_condition=filter_condition,
        filter_days=filter_days,
        date_range_text=date_range_text if 'date_range_text' in locals() else "",
        settings_message=settings_message,
        current_user=current_user,
        db_url_display=db_url_display,
        dataset_stats=dataset_stats,
        dataset_images=dataset_images,
        is_admin=is_admin,
        healthy_only_mode=HEALTHY_ONLY_MODE
    )


@app.route("/report")
def report():
    gate = require_login()
    if gate is not None:
        return gate
        
    username = session.get("user")
    current_user_id = None
    is_admin = (username == "admin")
    
    if username and not is_admin:
        try:
            user = get_user_by_username(username)
            if user:
                current_user_id = user.get("id")
        except:
            pass
            
    # Parse filters
    condition = request.args.get("condition")
    time_range = request.args.get("range")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    
    # Map time range to days or custom dates
    days = None
    filter_start = None
    filter_end = None
    
    range_label = "All Time"
    
    if time_range == "day":
        days = 1
        range_label = "Last 24 Hours"
    elif time_range == "week":
        days = 7
        range_label = "Last 7 Days"
    elif time_range == "month":
        days = 30
        range_label = "Last 30 Days"
    elif time_range == "custom":
        if start_date:
            filter_start = start_date
        if end_date:
            filter_end = end_date
        range_label = f"{start_date or '?'} to {end_date or '?'}"
            
    # Fetch data for report
    filter_user_id = None if is_admin else current_user_id
    try:
        limit = 1000  # Higher limit for reports
        
        if condition in ["Healthy", "Diseased"]:
            results = get_detection_results_by_condition(
                condition=condition,
                user_id=filter_user_id,
                days=days,
                start_date=filter_start,
                end_date=filter_end
            )
            # Apply limit manually since by_condition doesn't support limit in current db.py signature (oops, I should have checked)
            # Actually I didn't add limit to get_detection_results_by_condition signature. 
            # But the response is a list, so I can slice it.
            results = results[:limit]
        else:
            results = get_all_detection_results(
                limit=limit,
                user_id=filter_user_id,
                days=days,
                start_date=filter_start,
                end_date=filter_end
            )
            
        # Calculate stats from filtered results
        total_scans = len(results)
        healthy_count = sum(1 for r in results if r.get("condition") == "Healthy")
        diseased_count = sum(1 for r in results if r.get("condition") == "Diseased")
        
        if total_scans > 0:
            total_confidence = sum(
                float(r.get("confidence", 0)) for r in results if r.get("confidence")
            )
            success_rate = round(total_confidence / total_scans)
        else:
            success_rate = 0
            
        stats = {
            "total_scans": total_scans,
            "healthy_leaves": healthy_count,
            "diseased_leaves": diseased_count,
            "success_rate": success_rate
        }
        
    except Exception as e:
        results = []
        stats = {}
        print(f"Error fetching report data: {e}")
        import traceback
        traceback.print_exc()
    
    return render_template(
        "report.html", 
        results=results, 
        stats=stats, 
        username=username, 
        date=datetime.now(),
        filters={
            "condition": condition or "All",
            "range": range_label
        }
    )


@app.get("/delete")
def delete_file():
    gate = require_login()
    if gate is not None:
        return gate
    filename = request.args.get("file", "")
    safe_name = os.path.basename(filename)
    fpath = os.path.join(UPLOAD_FOLDER, safe_name)
    
    # Get current user ID for permission check
    username = session.get("user")
    current_user_id = None
    is_admin = (username == "admin")
    
    if username and not is_admin:
        try:
            user = get_user_by_username(username)
            if user:
                current_user_id = user.get("id")
        except Exception as e:
            print(f"Error fetching current user: {e}")
    
    try:
        # Delete from Supabase database first (with permission check)
        # Admin can delete any result, regular users can only delete their own
        try:
            delete_detection_result(safe_name, user_id=None if is_admin else current_user_id)
        except DatabaseError as e:
            flash(f"Error: {str(e)}")
            return redirect(url_for("dashboard", page="results"))
        except Exception as e:
            print(f"Error deleting from database: {e}")
            flash(f"Error deleting from database: {str(e)}")
            return redirect(url_for("dashboard", page="results"))
        
        # Delete file from filesystem
        if os.path.exists(fpath):
            os.remove(fpath)
        
        flash("File deleted successfully.")
    except OSError as e:
        flash(f"Error deleting file: {str(e)}")
    except Exception as e:
        flash(f"Error: {str(e)}")
    
    return redirect(url_for("dashboard", page="results"))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    # Bind to all interfaces so other devices (ESP32/phones) can reach it
    # Disable template caching in debug mode
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


