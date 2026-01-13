import os
import glob
from dotenv import load_dotenv
from supabase import create_client, Client
import json
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url: str = os.getenv("SUPABASE_URL", "")
supabase_key: str = os.getenv("SUPABASE_ANON_KEY", "")

supabase: Client = None
if supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"⚠️ Failed to connect to Supabase: {e}")

# Configuration
# Use environment variable DATASET_DIRS if available (semicolon separated), otherwise default to current dir or known paths
env_dataset_dirs = os.getenv("DATASET_DIRS", "")
if env_dataset_dirs:
    DATASET_DIRS = [d.strip() for d in env_dataset_dirs.split(";") if d.strip()]
else:
    # Default to current directory and common locations if not specified
    DATASET_DIRS = [
        os.path.join(os.getcwd(), "Dataset", "Healthy Pechay.v1i.yolov9"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "Dataset", "Healthy Pechay.v1i.yolov9")),
    ]

# Ensure directories exist
DATASET_DIRS = [d for d in DATASET_DIRS if os.path.exists(d)]
if not DATASET_DIRS:
    print("Warning: No valid dataset directories found. Please set DATASET_DIRS environment variable.")

def _collect_images(base_dir: str):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_files = []
    for root, _dirs, files in os.walk(base_dir):
        for file in files:
            lower = file.lower()
            if not any(lower.endswith(ext) for ext in image_extensions):
                continue
            dataset_type = "unknown"
            root_lower = root.lower()
            if "train" in root_lower:
                dataset_type = "train"
            elif "valid" in root_lower or "val" in root_lower:
                dataset_type = "val"
            elif "test" in root_lower:
                dataset_type = "test"
            all_files.append({
                "filename": file,
                "file_type": "image",
                "dataset_type": dataset_type,
                "url": os.path.join(root, file).replace("\\", "/")
            })
    return all_files

def scan_yolo_dataset(dataset_dir: str, batch_size: int = 16, conf: float = 0.25):
    if not os.path.exists(dataset_dir):
        print(f"⚠️ Directory not found: {dataset_dir}")
        return None

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"❌ Ultralytics YOLO not available: {e}")
        return None

    weight_candidates = [
        r"C:\Users\Admin\Documents\upload_yolo_supabase\petchay_detection\petchay_model\weights\best.pt",
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "petchay_detection", "petchay_model", "weights", "best.pt"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "weights", "best.pt"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "best.pt"),
    ]
    weight_path = None
    for p in weight_candidates:
        if os.path.exists(p):
            weight_path = p
            break
    if not weight_path:
        print("❌ best.pt not found in known locations")
        return None

    model = YOLO(weight_path)
    images = _collect_images(dataset_dir)
    image_paths = [item["url"] for item in images]
    total = len(image_paths)
    print(f"Scanning {total} images with YOLO: {dataset_dir}")
    print(f"Using weights: {weight_path}")
    print(f"YOLO names: {getattr(model, 'names', None)}")

    detected = 0
    no_det = 0
    label_counts = Counter()
    dataset_split_counts = Counter()
    
    # Store features for similarity matching
    # Since we can't easily extract embeddings with YOLO alone without modification,
    # we will rely on color/texture features calculated by app.py later.
    # But we should ensure we have the file list indexed.
    
    # However, if we want "accuracy", we might want to store some metadata here.
    # For now, just ensuring the file list is correct is step 1.

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_items = images[i:i+batch_size]
        try:
            results = model.predict(batch_paths, verbose=False, conf=conf)
        except TypeError:
            results = model.predict(batch_paths, verbose=False)

        for r, meta in zip(results, batch_items):
            boxes = getattr(r, "boxes", None)
            if not boxes or len(boxes) == 0:
                no_det += 1
                dataset_split_counts[f"{meta['dataset_type']}_no_det"] += 1
                continue
            detected += 1
            dataset_split_counts[f"{meta['dataset_type']}_det"] += 1

            try:
                confs = boxes.conf
                clss = boxes.cls
                conf_list = confs.cpu().tolist() if hasattr(confs, "cpu") else list(confs)
                cls_list = clss.cpu().tolist() if hasattr(clss, "cpu") else list(clss)
                best_conf = max(conf_list) if conf_list else 0.0
                best_idx = conf_list.index(best_conf) if conf_list else 0
                best_cls = int(cls_list[best_idx]) if cls_list else 0
            except Exception:
                best_cls = 0

            names = getattr(r, "names", None) or getattr(model, "names", None) or {}
            label = names.get(best_cls, str(best_cls)) if isinstance(names, dict) else str(best_cls)
            label_counts[str(label)] += 1

        done = min(i + batch_size, total)
        print(f"Processed {done}/{total}", end="\r")

    print()
    print(f"Total images: {total}")
    print(f"Detected (has boxes): {detected}")
    print(f"No detections: {no_det}")
    if label_counts:
        print("Top predicted labels:")
        for label, cnt in label_counts.most_common(10):
            print(f"- {label}: {cnt}")
    if dataset_split_counts:
        print("Split stats:")
        for k, v in sorted(dataset_split_counts.items()):
            print(f"- {k}: {v}")

    return {
        "dataset_dir": dataset_dir,
        "total": total,
        "detected": detected,
        "no_detections": no_det,
        "label_counts": dict(label_counts),
        "split_counts": dict(dataset_split_counts),
        "weights": weight_path,
        "names": getattr(model, "names", None),
    }

def populate_yolo_files():
    """
    Scans dataset directories and populates the yolo_files table 
    (Supabase and local offline storage)
    """
    print("Starting population of yolo_files...")
    
    count = 0
    errors = 0
    
    all_files = []
    for base_dir in DATASET_DIRS:
        if not os.path.exists(base_dir):
            print(f"⚠️ Directory not found: {base_dir}")
            continue
        print(f"Scanning {base_dir}...")
        all_files.extend(_collect_images(base_dir))

    print(f"Found {len(all_files)} images in datasets.")
    
    # 2. Insert into Supabase
    if supabase:
        print("Inserting into Supabase yolo_files table...")
        # Batch insert to avoid timeouts
        batch_size = 100
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            try:
                # Remove 'url' if it's a local absolute path, keep filename relative if possible
                # But schema says 'url' text null, so we can put anything. 
                # Let's keep it clean: just filename and dataset_type are critical for matching.
                
                # Check if table exists by trying to select 1 row
                # If error, try to create table via SQL RPC if possible, otherwise we can't do much from here without admin API
                
                clean_batch = []
                for item in batch:
                    clean_batch.append({
                        "filename": item["filename"],
                        "file_type": item["file_type"],
                        "dataset_type": item["dataset_type"],
                        "url": f"dataset/{item['dataset_type']}/{item['filename']}" # Mock URL
                    })
                
                # supabase.table("yolo_files").upsert(clean_batch, on_conflict="filename").execute()
                # Fallback to insert since upsert failed due to missing constraint
                supabase.table("yolo_files").insert(clean_batch).execute()
                count += len(batch)
                print(f"Processed {count}/{len(all_files)}...", end="\r")
            except Exception as e:
                print(f"\n❌ Error inserting batch: {e}")
                errors += 1
                
                # If table doesn't exist, we might get an error here.
                if "relation" in str(e) and "does not exist" in str(e):
                    print("\n⚠️ Table 'yolo_files' does not exist. Please run the SQL provided by the user.")
                    break

    # 3. Update local db.py offline storage (mock)
    # Since db.py uses in-memory _offline_dataset list which isn't persisted to disk,
    # we need to persist this list to a file so app.py can load it next time.
    # However, app.py/db.py doesn't seem to load from a file on startup for offline_dataset.
    # We should create a JSON file 'yolo_files.json' and modify db.py to load from it.
    
    yolo_files_local = [
        {
            "filename": item["filename"],
            "file_type": item["file_type"],
            "dataset_type": item["dataset_type"],
            "url": item["url"]
        }
        for item in all_files
    ]
    
    with open("yolo_files.json", "w") as f:
        json.dump(yolo_files_local, f, indent=2)
    print(f"\nSaved {len(yolo_files_local)} entries to local yolo_files.json")
    
    print("\nDone!")
    print(f"Total processed: {count}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    import sys
    raw_args = [a.strip() for a in sys.argv[1:] if a.strip()]
    lower_args = [a.lower() for a in raw_args]
    scan_mode = ("scan" in lower_args) or ("--scan" in lower_args)
    scan_dirs = [a for a in raw_args if a.lower() not in ("scan", "--scan")]
    if scan_mode:
        dirs = scan_dirs if scan_dirs else DATASET_DIRS
        for d in dirs:
            scan_yolo_dataset(d)
    else:
        populate_yolo_files()
