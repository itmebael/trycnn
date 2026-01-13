
import os
import shutil
import yaml
import threading
import logging
from ultralytics import YOLO
from db import get_dataset_entries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global lock to prevent concurrent training
TRAINING_LOCK = threading.Lock()

def train_yolo_model(base_dir):
    """
    Trains the YOLO model using data from the database.
    This function:
    1. Prepares a YOLO-format dataset from DB entries (Auto-Labeling).
    2. Trains the model (Object Detection with full-image boxes).
    3. Saves the best model to weights/best.pt.
    """
    if not TRAINING_LOCK.acquire(blocking=False):
        logger.warning("Training already in progress. Skipping.")
        return False

    try:
        logger.info("Starting YOLO training pipeline...")
        
        # 1. Prepare Data Directory
        dataset_root = os.path.join(base_dir, "yolo_dataset")
        
        # Clean up old dataset build
        if os.path.exists(dataset_root):
            try:
                shutil.rmtree(dataset_root)
            except Exception as e:
                logger.error(f"Failed to clean dataset dir: {e}")
                # Continue anyway, might be locked files
        
        images_train = os.path.join(dataset_root, "images", "train")
        images_val = os.path.join(dataset_root, "images", "val")
        labels_train = os.path.join(dataset_root, "labels", "train")
        labels_val = os.path.join(dataset_root, "labels", "val")
        
        for d in [images_train, images_val, labels_train, labels_val]:
            os.makedirs(d, exist_ok=True)
            
        # 2. Fetch Data
        entries = get_dataset_entries()
        
        # Fallback: Try loading from yolo_files.json if DB is empty
        if not entries and os.path.exists("yolo_files.json"):
            logger.info("Database empty. Falling back to yolo_files.json...")
            import json
            try:
                with open("yolo_files.json", "r") as f:
                    yolo_data = json.load(f)
                    
                for item in yolo_data:
                    path = item.get("url", "")
                    if not path: continue
                    
                    # Simple heuristic labeling based on file path/name
                    lower_path = path.lower()
                    condition = "Diseased"
                    disease = "Diseased" # Default
                    
                    if "healthy" in lower_path:
                        condition = "Healthy"
                        disease = None
                    else:
                        # Try to identify specific diseases
                        if "alternaria" in lower_path: disease = "Alternaria"
                        elif "black" in lower_path and "rot" in lower_path: disease = "Black Rot"
                        elif "soft" in lower_path and "rot" in lower_path: disease = "Soft Rot"
                        elif "leaf" in lower_path and "spot" in lower_path: disease = "Leaf Spot"
                    
                    entries.append({
                        "image_url": path,
                        "condition": condition,
                        "disease_name": disease
                    })
                logger.info(f"Loaded {len(entries)} entries from yolo_files.json")
            except Exception as e:
                logger.error(f"Failed to load yolo_files.json: {e}")

        if not entries:
            logger.warning("No data found in database or yolo_files.json to train on.")
            return False

        # 3. Analyze Classes
        # Logic: Condition (Healthy/Diseased) + Disease Name
        class_names = set()
        for entry in entries:
            condition = entry.get("condition", "Healthy")
            disease = entry.get("disease_name")
            
            if condition == "Healthy":
                label = "Healthy"
            else:
                label = disease if disease else "Diseased"
            class_names.add(label)
            
        # Sort to ensure consistent ID mapping
        sorted_classes = sorted(list(class_names))
        class_map = {name: i for i, name in enumerate(sorted_classes)}
        
        logger.info(f"Training with classes: {class_map}")
        
        # 4. Split Data (80/20)
        import random
        # Filter valid entries first
        valid_entries = []
        for entry in entries:
            url = entry.get("image_url", "")
            
            # Check if absolute path exists directly
            if os.path.exists(url):
                entry['_local_path'] = url
                valid_entries.append(entry)
                continue
                
            # Check if it works with os.path.normpath (fixes separators)
            norm_url = os.path.normpath(url)
            if os.path.exists(norm_url):
                entry['_local_path'] = norm_url
                valid_entries.append(entry)
                continue

            # Legacy logic for "uploads/" relative paths
            if "uploads" in url:
                rel_path = url.lstrip("/")
                # Handle leading slash removal carefully
                if rel_path.startswith("/"): rel_path = rel_path[1:]
                
                # Check absolute path relative to base_dir
                possible_path = os.path.join(base_dir, rel_path)
                # Also try replacing forward slashes with os.sep
                possible_path_os = possible_path.replace("/", os.sep)
                
                if os.path.exists(possible_path_os):
                    entry['_local_path'] = possible_path_os
                    valid_entries.append(entry)
                elif os.path.exists(possible_path):
                    entry['_local_path'] = possible_path
                    valid_entries.append(entry)
        
        if not valid_entries:
            logger.warning("No valid local images found for training.")
            return False

        random.shuffle(valid_entries)
        split_idx = int(len(valid_entries) * 0.8)
        train_entries = valid_entries[:split_idx]
        val_entries = valid_entries[split_idx:]
        
        # 5. Process Entries (Copy Images & Create Labels)
        def process_batch(entry_list, img_dir, lbl_dir):
            count = 0
            for entry in entry_list:
                img_path = entry['_local_path']
                
                # Determine class ID
                condition = entry.get("condition", "Healthy")
                disease = entry.get("disease_name")
                label_name = "Healthy" if condition == "Healthy" else (disease if disease else "Diseased")
                class_id = class_map.get(label_name, 0)
                
                # Copy image
                filename = os.path.basename(img_path)
                dest_img_path = os.path.join(img_dir, filename)
                shutil.copy(img_path, dest_img_path)
                
                # Create label file (Full Image Box)
                # class_id center_x center_y width height
                txt_name = os.path.splitext(filename)[0] + ".txt"
                with open(os.path.join(lbl_dir, txt_name), "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                count += 1
            return count

        n_train = process_batch(train_entries, images_train, labels_train)
        n_val = process_batch(val_entries, images_val, labels_val)
        logger.info(f"Prepared dataset: {n_train} train, {n_val} val images.")
        
        # 6. Create dataset.yaml
        yaml_path = os.path.join(dataset_root, "dataset.yaml")
        yaml_data = {
            "path": dataset_root,
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(sorted_classes)}
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
            
        # 7. Train Model
        weights_dir = os.path.join(base_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        best_weights = os.path.join(weights_dir, "best.pt")
        
        # Start from existing best.pt if available to incremental learn (sort of), 
        # or yolov8n.pt for fresh start. 
        # Note: YOLOv8 doesn't truly support incremental learning easily without freezing, 
        # but starting from weights helps.
        start_weights = "yolov8n.pt"
        if os.path.exists(best_weights):
            start_weights = best_weights
            
        logger.info(f"Loading model {start_weights}...")
        model = YOLO(start_weights)
        
        logger.info("Starting training process (this may take a while)...")
        # Train
        results = model.train(
            data=yaml_path, 
            epochs=10, 
            imgsz=640,
            project=os.path.join(base_dir, "runs"),
            name="petchay_yolo",
            exist_ok=True,
            verbose=True
        )
        
        # 8. Save Result
        trained_weights = os.path.join(base_dir, "runs", "petchay_yolo", "weights", "best.pt")
        if os.path.exists(trained_weights):
            # Backup old weights
            if os.path.exists(best_weights):
                shutil.copy(best_weights, best_weights + ".bak")
                
            shutil.copy(trained_weights, best_weights)
            logger.info(f"Training success! Model saved to {best_weights}")
            return True
        else:
            logger.error("Training finished but best.pt not found.")
            return False

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        TRAINING_LOCK.release()

if __name__ == "__main__":
    # Get the base directory (where this script is located)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running YOLO training manually in {base_dir}...")
    
    success = train_yolo_model(base_dir)
    
    if success:
        print("Training completed successfully!")
    else:
        print("Training failed or was skipped.")
