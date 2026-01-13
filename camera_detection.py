import cv2
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Try to import Ultralytics YOLO (as requested)
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: 'ultralytics' not installed. Running in compatibility mode with existing CNN.")

# Import local modules
from db import supabase, upload_image_to_storage, create_detection_result
from predict import PechayPredictor

# Load environment variables
load_dotenv()

# Configuration
YOLO_MODEL_PATH = os.getenv("YOLO_WEIGHTS_PATH", "petchay_detection/petchay_model/weights/best.pt")
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", "pechay_cnn_model_20251212_184656.pth") # Adjust to your latest model
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "predictions"))
BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", "petchay-images")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Model
model = None
predictor_type = "CNN" # Default to CNN if YOLO fails

if HAS_YOLO and os.path.exists(YOLO_MODEL_PATH):
    print(f"Loading YOLOv9 model from {YOLO_MODEL_PATH}...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        predictor_type = "YOLO"
        print("✅ YOLOv9 model loaded!")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")

if model is None:
    print(f"Falling back to existing CNN model from {CNN_MODEL_PATH}...")
    if os.path.exists(CNN_MODEL_PATH):
        try:
            model = PechayPredictor(CNN_MODEL_PATH)
            predictor_type = "CNN"
            print("✅ CNN model loaded!")
        except Exception as e:
            print(f"Failed to load CNN model: {e}")
            print("❌ No valid model found. Exiting.")
            exit(1)
    else:
        print(f"❌ CNN model file not found at {CNN_MODEL_PATH}")
        # Try to find any .pth file
        import glob
        pth_files = glob.glob("*.pth")
        if pth_files:
            CNN_MODEL_PATH = pth_files[0]
            print(f"Found alternative model: {CNN_MODEL_PATH}")
            model = PechayPredictor(CNN_MODEL_PATH)
            predictor_type = "CNN"
        else:
            print("❌ No model files found.")
            exit(1)

def process_frame(frame, save_prefix="frame"):
    """
    Process a single frame/image: detect, save, upload.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_prefix}_{timestamp}.jpg"
    local_path = os.path.join(OUTPUT_DIR, filename)
    
    # Save frame temporarily for processing
    cv2.imwrite(local_path, frame)
    
    results_data = {}
    
    # 1. Run Detection
    if predictor_type == "YOLO":
        results = model.predict(local_path)
        # Save annotated image
        annotated_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
        results[0].save(annotated_path) # YOLO saves to its own dir usually, but let's assume this works or we use plot()
        
        # YOLO result parsing (simplified)
        # We would extract class, conf, bbox here
        # For now, just using the image
        final_image_path = annotated_path
        
        # Display
        cv2.imshow("Pechay Detection (YOLO)", cv2.imread(annotated_path))
        
    else: # CNN
        # CNN doesn't draw boxes, so we just use the original frame
        # But we can overlay text
        result = model.predict_image(local_path)
        
        condition = result['condition']
        confidence = result['confidence']
        
        # Draw on frame
        display_frame = frame.copy()
        color = (0, 255, 0) if condition == "Healthy" else (0, 0, 255)
        text = f"{condition} ({confidence}%)"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Pechay Detection (CNN)", display_frame)
        final_image_path = local_path
        
        results_data = {
            "condition": condition,
            "confidence": confidence
        }

    # 2. Upload to Supabase Storage
    print(f"Uploading {filename} to Supabase...")
    public_url = upload_image_to_storage(final_image_path, BUCKET_NAME)
    
    if public_url:
        print(f"✅ Uploaded: {public_url}")
        
        # 3. Save Metadata to DB
        # If we have a logged in user, we'd use their ID, but here we might be anonymous or admin
        try:
            create_detection_result(
                filename=os.path.basename(final_image_path),
                condition=results_data.get("condition", "Unknown"),
                confidence=results_data.get("confidence", 0),
                image_path=final_image_path, # Local path or URL? DB expects local usually but let's see
                recommendations={},
                all_probabilities={},
                user_id=None # Anonymous/System
            )
            print("✅ Metadata saved to DB")
        except Exception as e:
            print(f"⚠️ Metadata save failed: {e}")

def detect_camera():
    print("Starting Camera... Press 'q' to quit, 'c' to capture & detect.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow("Live Feed (Press 'c' to capture)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Capturing...")
            process_frame(frame, "cam")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Select mode:")
    print("1. Camera Detection")
    print("2. Image Upload (Test)")
    mode = input("Enter 1 or 2: ")
    
    if mode == "1":
        detect_camera()
    elif mode == "2":
        path = input("Enter image path: ").strip('"')
        if os.path.exists(path):
            img = cv2.imread(path)
            process_frame(img, "upload")
        else:
            print("File not found.")
