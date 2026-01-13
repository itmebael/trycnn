
import os
from typing import Optional, List, Dict, Any
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = object
from dotenv import load_dotenv
from datetime import datetime
import json
from werkzeug.security import generate_password_hash

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url: str = os.getenv("SUPABASE_URL", "")
supabase_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "")

# In-memory storage for offline mode
_offline_users = {}
_offline_results = []
_offline_dataset = []
_custom_training_data = []

# Try to load offline data from JSON files
try:
    if os.path.exists("yolo_files.json"):
        with open("yolo_files.json", "r") as f:
            _loaded_data = json.load(f)
            if isinstance(_loaded_data, list):
                _offline_dataset.extend(_loaded_data)
                print(f"Loaded {len(_loaded_data)} entries from yolo_files.json for offline mode")
except Exception as e:
    print(f"Error loading yolo_files.json: {e}")

try:
    if os.path.exists("custom_dataset.json"):
        with open("custom_dataset.json", "r") as f:
            _loaded_custom = json.load(f)
            if isinstance(_loaded_custom, list):
                _custom_training_data.extend(_loaded_custom)
                print(f"Loaded {len(_loaded_custom)} entries from custom_dataset.json")
except Exception as e:
    print(f"Error loading custom_dataset.json: {e}")

# Initialize Supabase with connectivity check
supabase: Optional[Client] = None

if SUPABASE_AVAILABLE and supabase_url and supabase_key:
    try:
        print(f"Attempting to connect to Supabase at {supabase_url}...")
        temp_client = create_client(supabase_url, supabase_key)
        
        # Test connection with a lightweight query (fetch 1 user or just check health)
        # We use a short timeout logic if possible, but here we just try a query.
        # If DNS fails, this raises an exception immediately.
        try:
            # We don't need the result, just checking if it raises an error
            # Using 'users' table as it's core to the app
            temp_client.table("users").select("count", count="exact").limit(1).execute()
            print("Supabase connection successful.")
            supabase = temp_client
        except Exception as conn_err:
            print(f"Supabase connection test failed: {conn_err}")
            print("Switching to OFFLINE MODE.")
            supabase = None
            
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        supabase = None
else:
    if not SUPABASE_AVAILABLE:
        print("Warning: Supabase library not found.")
    elif not supabase_url or not supabase_key:
        print("Warning: Missing Supabase credentials.")
    print("Running in OFFLINE MODE.")
    supabase = None

# If offline, ensure admin user exists
if not supabase:
    # Check if admin already exists in _offline_users (unlikely on startup)
    if "admin" not in _offline_users:
        admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
        print(f"Creating offline admin user (password: {admin_pass})")
        _offline_users["admin"] = {
            "id": "admin_offline",
            "username": "admin",
            "email": "admin@local.test",
            "password": generate_password_hash(admin_pass),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "role": "admin"
        }


def upload_image_to_storage(file_path: str, bucket_name: str = "petchay-images") -> Optional[str]:
    """Uploads a file to Supabase Storage and returns the public URL."""
    if not supabase:
        return None
    try:
        file_name = os.path.basename(file_path)
        
        # Determine content type from file extension
        ext = os.path.splitext(file_name)[1].lower()
        content_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        content_type = content_type_map.get(ext, 'image/jpeg')
        
        with open(file_path, "rb") as f:
            file_data = f.read()
            # Upload with content type specified
            # Use a subfolder structure: dataset/{filename} for dataset images
            # This keeps scan images and dataset images organized
            storage_path = file_name
            if "dataset_custom" in file_path:
                # Extract folder name from path (e.g., dataset_custom/Healthy/filename.jpg)
                path_parts = file_path.replace("\\", "/").split("/")
                if "dataset_custom" in path_parts:
                    idx = path_parts.index("dataset_custom")
                    if idx + 1 < len(path_parts):
                        folder = path_parts[idx + 1]  # Get folder name (Healthy, Diseased, etc.)
                        storage_path = f"dataset/{folder}/{file_name}"
            
            print(f"[STORAGE] Uploading {file_name} to {bucket_name}/{storage_path}")
            try:
                supabase.storage.from_(bucket_name).upload(
                    storage_path, 
                    file_data,
                    file_options={"content-type": content_type, "upsert": "true"}
                )
                print(f"[STORAGE] ✅ Upload successful to {storage_path}")
            except Exception as upload_err:
                print(f"[STORAGE] ❌ Upload failed: {upload_err}")
                print(f"[STORAGE]   Bucket: {bucket_name}")
                print(f"[STORAGE]   Path: {storage_path}")
                raise upload_err
            
            # Update file_name for URL construction
            file_name = storage_path
        
        # Construct public URL
        if "//" in supabase_url:
             project_id = supabase_url.split("//")[1].split(".")[0]
        else:
             project_id = "unknown"
        public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{file_name}"
        print(f"[STORAGE] ✅ Public URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"[STORAGE] ❌ Error uploading to Supabase Storage: {e}")
        print(f"[STORAGE]   File: {file_path}")
        print(f"[STORAGE]   Bucket: {bucket_name}")
        import traceback
        traceback.print_exc()
        return None

class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user by username"""
    if not supabase:
        return _offline_users.get(username)
    try:
        response = supabase.table("users").select("*").eq("username", username).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        raise DatabaseError(f"Error fetching user: {str(e)}")

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    if not supabase:
        for user in _offline_users.values():
            if user.get("email") == email:
                return user
        return None
    try:
        response = supabase.table("users").select("*").eq("email", email).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        raise DatabaseError(f"Error fetching user: {str(e)}")

def create_user(username: str, email: str, password_hash: str) -> Dict[str, Any]:
    """Create a new user"""
    if not supabase:
        user_id = f"offline_user_{len(_offline_users) + 1}"
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "password": password_hash,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        _offline_users[username] = user_data
        return user_data
    try:
        user_data = {
            "username": username,
            "email": email,
            "password": password_hash
        }
        response = supabase.table("users").insert(user_data).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        raise DatabaseError("Failed to create user")
    except Exception as e:
        raise DatabaseError(f"Error creating user: {str(e)}")

def verify_user_credentials(username: str, password_hash: str):
    """Legacy function, now handled in app.py logic"""
    pass

def create_detection_result(
    filename: str,
    condition: str,
    confidence: float,
    image_path: str,
    recommendations: Dict[str, Any],
    all_probabilities: Optional[Dict[str, float]] = None,
    user_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    disease_name: Optional[str] = None,
    treatment: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new detection result"""
    if not supabase:
        result_data = {
            "filename": filename,
            "condition": condition,
            "confidence": float(confidence),
            "image_path": image_path,
            "recommendations": recommendations,
            "timestamp": timestamp or datetime.now().isoformat(),
            "id": f"offline_result_{len(_offline_results) + 1}",
            "user_id": user_id
        }
        if all_probabilities:
            result_data["all_probabilities"] = all_probabilities
        if disease_name:
            result_data["disease_name"] = disease_name
        if treatment:
            result_data["treatment"] = treatment
        _offline_results.append(result_data)
        return result_data
    try:
        result_data = {
            "filename": filename,
            "condition": condition,
            "confidence": float(confidence),
            "image_path": image_path,
            "recommendations": recommendations,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        if all_probabilities:
            result_data["all_probabilities"] = all_probabilities
        if user_id:
            result_data["user_id"] = user_id
        if disease_name:
            result_data["disease_name"] = disease_name
        if treatment:
            result_data["treatment"] = treatment
        response = supabase.table("detection_results").insert(result_data).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        raise DatabaseError("Failed to create detection result")
    except Exception as e:
        raise DatabaseError(f"Error creating detection result: {str(e)}")

def get_detection_result_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Get detection result by filename"""
    if not supabase:
        for result in reversed(_offline_results):
            if result.get("filename") == filename:
                return result
        return None
    try:
        response = supabase.table("detection_results").select("*").eq("filename", filename).order("timestamp", desc=True).limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    except Exception as e:
        raise DatabaseError(f"Error fetching detection result: {str(e)}")

def get_all_detection_results(limit: int = 100, offset: int = 0, user_id: Optional[str] = None, days: Optional[int] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all detection results with pagination and filters"""
    if not supabase:
        filtered = _offline_results
        if user_id:
            filtered = [r for r in filtered if r.get("user_id") == user_id]
        if days:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=days)
            filtered = [r for r in filtered if r.get("timestamp") >= cutoff.isoformat()]
        return filtered[-limit:]
    try:
        query = supabase.table("detection_results").select("*").order("timestamp", desc=True)
        if user_id:
            query = query.eq("user_id", user_id)
        if days:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query = query.gte("timestamp", cutoff)
        response = query.range(offset, offset + limit - 1).execute()
        return response.data
    except Exception as e:
        raise DatabaseError(f"Error fetching detection results: {str(e)}")

def get_dashboard_stats(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get dashboard statistics"""
    if not supabase:
        results = _offline_results
        if user_id:
            results = [r for r in results if r.get("user_id") == user_id]
        total = len(results)
        healthy = sum(1 for r in results if r.get("condition") == "Healthy")
        diseased = sum(1 for r in results if r.get("condition") == "Diseased")
        success_rate = int((healthy / total * 100)) if total > 0 else 0
        return {
            "total_scans": total,
            "healthy_leaves": healthy,
            "diseased_leaves": diseased,
            "success_rate": success_rate
        }
    try:
        # Optimised: just get total
        # Get healthy count
        q_healthy = supabase.table("detection_results").select("id", count="exact").eq("condition", "Healthy")
        if user_id:
            q_healthy = q_healthy.eq("user_id", user_id)
        healthy = q_healthy.execute().count
        
        # Get diseased count
        q_diseased = supabase.table("detection_results").select("id", count="exact").eq("condition", "Diseased")
        if user_id:
            q_diseased = q_diseased.eq("user_id", user_id)
        diseased = q_diseased.execute().count
        
        total = healthy + diseased # Approximation, or query total
        success_rate = int((healthy / total * 100)) if total > 0 else 0
        return {
            "total_scans": total,
            "healthy_leaves": healthy,
            "diseased_leaves": diseased,
            "success_rate": success_rate
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {"total_scans": 0, "healthy_leaves": 0, "diseased_leaves": 0, "success_rate": 0}

# Placeholder functions for imports that might exist in app.py
def get_dataset_images(): return []
def create_dataset_image(): pass
def get_dataset_stats(): return {}
def create_yolo_file(
    filename: str,
    file_type: str = "image",
    dataset_type: str = None,
    url: str = None,
    treatment: str = None
) -> Dict[str, Any]:
    """
    Create a new entry in the yolo_files table.
    Used for tracking files that are part of the YOLO training dataset.
    """
    data_entry = {
        "filename": filename,
        "file_type": file_type,
        "dataset_type": dataset_type,
        "url": url or filename,
        "treatment": treatment,
        "uploaded_at": datetime.now().isoformat()
    }
    
    # 1. Save to Supabase if available
    if supabase:
        try:
            response = supabase.table("yolo_files").insert(data_entry).execute()
            if response.data:
                print(f"Saved yolo_file to Supabase: {filename}")
                return response.data[0] if response.data else data_entry
        except Exception as e:
            print(f"Error saving to Supabase yolo_files: {e}")
            # Continue to save locally as fallback
    
    # 2. Save to local JSON (Offline / Backup)
    _offline_dataset.append(data_entry)
    try:
        # Load existing data
        existing_data = []
        if os.path.exists("yolo_files.json"):
            with open("yolo_files.json", "r") as f:
                existing_data = json.load(f)
        
        # Add new entry
        existing_data.append(data_entry)
        
        # Save back to file
        with open("yolo_files.json", "w") as f:
            json.dump(existing_data, f, indent=2)
    except Exception as e:
        print(f"Error saving yolo_files.json: {e}")
    
    return data_entry
def create_file_upload_log(filename, file_path, original_name, file_size, upload_source, user_id): pass
def update_user_email(username, email): pass
def update_user_password(username, password_hash): pass
def delete_user(username): pass
def update_detection_result(): pass
def delete_detection_result(): pass
def get_detection_results_by_condition(): return []
def get_yolo_file_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Get a yolo_file entry by filename.
    Used during scanning to check if an image is in the training dataset.
    """
    if not filename:
        return None
    
    # 1. Check Supabase
    if supabase:
        try:
            response = supabase.table("yolo_files").select("*").eq("filename", filename).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
        except Exception as e:
            print(f"Error querying Supabase yolo_files: {e}")
    
    # 2. Check local offline storage
    try:
        if os.path.exists("yolo_files.json"):
            with open("yolo_files.json", "r") as f:
                yolo_files = json.load(f)
                for entry in yolo_files:
                    if entry.get("filename") == filename:
                        return entry
    except Exception as e:
        print(f"Error reading yolo_files.json: {e}")
    
    # 3. Check in-memory storage
    for entry in _offline_dataset:
        if entry.get("filename") == filename:
            return entry
    
    return None


def get_yolo_files_by_condition(condition: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get yolo_files entries filtered by condition (Healthy/Diseased) or all entries.
    Used for comparing uploaded images with database.
    """
    results = []
    
    # 1. Check Supabase
    if supabase:
        try:
            query = supabase.table("yolo_files").select("*")
            if condition:
                # Filter by dataset_type (which contains Healthy/Diseased/disease name)
                query = query.or_(f"dataset_type.eq.{condition},label.eq.{condition}")
            query = query.limit(limit).order("uploaded_at", desc=True)
            response = query.execute()
            if response.data:
                results.extend(response.data)
        except Exception as e:
            print(f"Error querying Supabase yolo_files: {e}")
    
    # 2. Check local offline storage
    try:
        if os.path.exists("yolo_files.json"):
            with open("yolo_files.json", "r") as f:
                yolo_files = json.load(f)
                for entry in yolo_files:
                    if condition:
                        dataset_type = entry.get("dataset_type", "")
                        label = entry.get("label", "")
                        if condition.lower() in str(dataset_type).lower() or condition.lower() in str(label).lower():
                            results.append(entry)
                    else:
                        results.append(entry)
                    if len(results) >= limit:
                        break
    except Exception as e:
        print(f"Error reading yolo_files.json: {e}")
    
    # 3. Check in-memory storage
    for entry in _offline_dataset:
        if condition:
            dataset_type = entry.get("dataset_type", "")
            label = entry.get("label", "")
            if condition.lower() in str(dataset_type).lower() or condition.lower() in str(label).lower():
                results.append(entry)
        else:
            results.append(entry)
        if len(results) >= limit:
            break
    
    return results[:limit]


def get_all_yolo_files(limit: int = 500) -> List[Dict[str, Any]]:
    """Get all yolo_files entries for comparison"""
    return get_yolo_files_by_condition(condition=None, limit=limit)

# New Functions for Hybrid Dataset Management
def save_dataset_entry(
    filename: str, 
    label: str, 
    image_url: str, 
    embedding: List[float], 
    disease_name: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save a new entry to the petchay_dataset table (and local JSON as backup).
    This supports the Hybrid Architecture (YOLO + Embeddings).
    """
    # Prepare data object
    data_entry = {
        "filename": filename,
        "condition": label,  # 'Healthy' or 'Diseased'
        "disease_name": disease_name or label,
        "image_url": image_url,
        "embedding": embedding,
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }

    # 1. Save to Supabase if available
    if supabase:
        try:
            # Note: 'embedding' column in Supabase should be 'vector' type or array
            # If using pgvector, it handles list automatically.
            response = supabase.table("petchay_dataset").insert(data_entry).execute()
            if response.data:
                print(f"Saved dataset entry to Supabase: {filename}")
        except Exception as e:
            print(f"Error saving to Supabase petchay_dataset: {e}")
            # Continue to save locally as fallback

    # 2. Save to local JSON (Offline / Backup)
    _custom_training_data.append(data_entry)
    try:
        with open("custom_dataset.json", "w") as f:
            json.dump(_custom_training_data, f)
    except Exception as e:
        print(f"Error saving custom_dataset.json: {e}")
        
    return data_entry

def get_dataset_entries() -> List[Dict[str, Any]]:
    """
    Retrieve all dataset entries for the Hybrid Matcher.
    Prioritizes Supabase, falls back to local JSON.
    """
    if supabase:
        try:
            # Fetch all entries (limit to 1000 for performance, or implement paging if needed)
            response = supabase.table("petchay_dataset").select("*").execute()
            if response.data:
                print(f"Loaded {len(response.data)} entries from Supabase")
                return response.data
        except Exception as e:
            print(f"Error loading from Supabase petchay_dataset: {e}")
    
    # Fallback to local
    return _custom_training_data

# Legacy wrapper for backward compatibility
def save_custom_training_data(filename: str, label: str, image_path: str, embedding: List[float], user_id: Optional[str] = None):
    return save_dataset_entry(filename, label, image_path, embedding, disease_name=label, user_id=user_id)

def get_custom_training_data() -> List[Dict[str, Any]]:
    return get_dataset_entries()
