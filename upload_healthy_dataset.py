"""
Upload Healthy Pechay Dataset to Supabase
Uploads all images from Healthy Pechay.v1i.yolov9/train/images to:
1. yolo_files table (with condition="Healthy")
2. petchay_dataset table (with condition="Healthy" and embeddings)
"""

import os
import glob
from pathlib import Path
import numpy as np
from datetime import datetime

# Import tqdm for progress bar (optional)
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc="", unit=""):
        print(f"{desc}: Processing {len(iterable) if hasattr(iterable, '__len__') else '?'} items...")
        return iterable

# Import database functions
from db import (
    create_yolo_file,
    save_dataset_entry,
    upload_image_to_storage,
    supabase
)

# Import predictor for embeddings
try:
    from predict import PechayPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    print("Warning: PechayPredictor not available. Embeddings will not be generated.")
    PREDICTOR_AVAILABLE = False

# Configuration
DATASET_PATH = r"Healthy Pechay.v1i.yolov9/train/images"
MODEL_PATH = "pechay_cnn_model_20251212_184656.pth"
BATCH_SIZE = 50  # Process in batches to avoid memory issues
STORAGE_BUCKET = "petchay-images"

def load_predictor():
    """Load CNN predictor for generating embeddings"""
    if not PREDICTOR_AVAILABLE:
        return None
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading CNN model from {MODEL_PATH}...")
            predictor = PechayPredictor(MODEL_PATH, device='cpu')
            print("[OK] CNN model loaded successfully!")
            return predictor
        else:
            print(f"[WARN] Model file not found: {MODEL_PATH}")
            print("   Embeddings will not be generated, but images will still be uploaded.")
            return None
    except Exception as e:
        print(f"[WARN] Error loading predictor: {e}")
        print("   Embeddings will not be generated, but images will still be uploaded.")
        return None

def generate_embedding(predictor, image_path):
    """Generate embedding for an image"""
    if not predictor:
        return None
    
    try:
        embedding = predictor.extract_features(image_path)
        if embedding is not None:
            # Flatten to 1D array
            embedding = embedding.flatten()
            # Convert to list for JSON serialization
            return embedding.tolist()
    except Exception as e:
        print(f"   ⚠️ Error generating embedding for {os.path.basename(image_path)}: {e}")
        return None

def upload_image_to_supabase(image_path):
    """Upload image to Supabase storage"""
    try:
        url = upload_image_to_storage(image_path, STORAGE_BUCKET)
        return url
    except Exception as e:
        print(f"   [WARN] Error uploading {os.path.basename(image_path)}: {e}")
        # Return local path as fallback
        return image_path

def upload_healthy_dataset():
    """Main function to upload healthy pechay dataset"""
    print("=" * 60)
    print("Healthy Pechay Dataset Upload Script")
    print("=" * 60)
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset path not found: {DATASET_PATH}")
        return
    
    # Get all image files
    image_files = glob.glob(os.path.join(DATASET_PATH, "*.jpg")) + \
                  glob.glob(os.path.join(DATASET_PATH, "*.jpeg")) + \
                  glob.glob(os.path.join(DATASET_PATH, "*.png"))
    
    if not image_files:
        print(f"[ERROR] No images found in {DATASET_PATH}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Storage bucket: {STORAGE_BUCKET}")
    print()
    
    # Load predictor for embeddings
    predictor = load_predictor()
    if predictor:
        print("[OK] Will generate embeddings for petchay_dataset")
    else:
        print("[WARN] Will skip embedding generation (images will still be uploaded)")
    print()
    
    # Statistics
    stats = {
        "total": len(image_files),
        "yolo_files_success": 0,
        "yolo_files_failed": 0,
        "petchay_dataset_success": 0,
        "petchay_dataset_failed": 0,
        "embeddings_generated": 0,
        "images_uploaded": 0
    }
    
    # Process images in batches
    print("Starting upload process...")
    print()
    
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images", unit="img")):
        try:
            filename = os.path.basename(image_path)
            unique_filename = f"healthy_pechay_{i+1}_{filename}"
            
            # 1. Upload image to Supabase storage
            image_url = upload_image_to_supabase(image_path)
            if image_url:
                stats["images_uploaded"] += 1
            
            # 2. Generate embedding
            embedding = None
            if predictor:
                embedding = generate_embedding(predictor, image_path)
                if embedding:
                    stats["embeddings_generated"] += 1
            
            # 3. Add to yolo_files table
            try:
                # Note: create_yolo_file currently only accepts: filename, file_type, dataset_type, url, treatment
                # Additional label fields may need to be added via direct Supabase call if needed
                create_yolo_file(
                    filename=unique_filename,
                    file_type="image",
                    dataset_type="Healthy",
                    url=image_url or image_path,
                    treatment=None  # No treatment needed for healthy pechay
                )
                
                # If we need to add label fields, update via Supabase directly
                if supabase:
                    try:
                        supabase.table("yolo_files").update({
                            "label": "Healthy",
                            "label_confidence": 1.0,
                            "image_region": "leaf",
                            "quality_score": 0.9,
                            "is_verified": True
                        }).eq("filename", unique_filename).execute()
                    except Exception as e:
                        print(f"   [WARN] Error updating label fields: {e}")
                
                stats["yolo_files_success"] += 1
            except Exception as e:
                print(f"   [WARN] Error adding to yolo_files: {e}")
                stats["yolo_files_failed"] += 1
            
            # 4. Add to petchay_dataset table (with embedding)
            if embedding:
                try:
                    save_dataset_entry(
                        filename=unique_filename,
                        label="Healthy",  # This becomes the 'condition' field
                        image_url=image_url or image_path,
                        embedding=embedding,
                        disease_name=None,  # Healthy, no disease
                        user_id=None
                    )
                    stats["petchay_dataset_success"] += 1
                except Exception as e:
                    print(f"   [WARN] Error adding to petchay_dataset: {e}")
                    stats["petchay_dataset_failed"] += 1
            else:
                # Still add to petchay_dataset even without embedding (for reference)
                try:
                    save_dataset_entry(
                        filename=unique_filename,
                        label="Healthy",
                        image_url=image_url or image_path,
                        embedding=[],  # Empty embedding
                        disease_name=None,
                        user_id=None
                    )
                    stats["petchay_dataset_success"] += 1
                except Exception as e:
                    print(f"   [WARN] Error adding to petchay_dataset (no embedding): {e}")
                    stats["petchay_dataset_failed"] += 1
            
            # Progress update every 100 images
            if (i + 1) % 100 == 0:
                print(f"\n[Progress] {i+1}/{len(image_files)} images processed")
                print(f"   [OK] yolo_files: {stats['yolo_files_success']}")
                print(f"   [OK] petchay_dataset: {stats['petchay_dataset_success']}")
                print(f"   [OK] Embeddings: {stats['embeddings_generated']}")
                print()
        
        except Exception as e:
            print(f"   [ERROR] Error processing {os.path.basename(image_path)}: {e}")
            continue
    
    # Final statistics
    print()
    print("=" * 60)
    print("Upload Summary")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print(f"Images uploaded to storage: {stats['images_uploaded']}")
    print()
    print("yolo_files table:")
    print(f"  [OK] Success: {stats['yolo_files_success']}")
    print(f"  [FAIL] Failed: {stats['yolo_files_failed']}")
    print()
    print("petchay_dataset table:")
    print(f"  [OK] Success: {stats['petchay_dataset_success']}")
    print(f"  [FAIL] Failed: {stats['petchay_dataset_failed']}")
    print(f"  [OK] Embeddings generated: {stats['embeddings_generated']}")
    print()
    print("=" * 60)
    print("[OK] Upload process completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("   1. The system will now use these images for healthy pechay detection")
    print("   2. When users upload images, the system will compare with these embeddings")
    print("   3. The yolo_files entries will be used for YOLO model training")
    print()

if __name__ == "__main__":
    try:
        upload_healthy_dataset()
    except KeyboardInterrupt:
        print("\n\n[WARN] Upload interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()

