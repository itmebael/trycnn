import os
from dotenv import load_dotenv
from supabase import create_client, Client
import mimetypes

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url: str = os.getenv("SUPABASE_URL", "")
supabase_key: str = os.getenv("SUPABASE_ANON_KEY", "")

if not supabase_url or not supabase_key:
    print("❌ Error: Missing Supabase credentials in .env")
    exit(1)

supabase: Client = create_client(supabase_url, supabase_key)

# Configuration
SOURCE_DIR = r"C:\Users\Admin\Downloads\petchaydataset.v1i.yolov9"
BUCKET_NAME = "yolo-dataset"

def upload_directory(source_dir, bucket_name):
    print(f"Starting upload from {source_dir} to bucket '{bucket_name}'...")
    
    # Check if bucket exists, create if not (if API allows, otherwise assume it exists)
    try:
        buckets = supabase.storage.list_buckets()
        bucket_exists = any(b.name == bucket_name for b in buckets)
        if not bucket_exists:
            print(f"Creating bucket '{bucket_name}'...")
            supabase.storage.create_bucket(bucket_name, options={"public": True})
    except Exception as e:
        print(f"⚠️ Warning checking/creating bucket: {e}")

    success_count = 0
    fail_count = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Calculate relative path for storage
            # e.g. C:\...\train\images\img.jpg -> train/images/img.jpg
            relative_path = os.path.relpath(file_path, source_dir)
            storage_path = relative_path.replace("\\", "/") # Ensure forward slashes
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = "application/octet-stream"
            
            print(f"Uploading {storage_path}...", end="", flush=True)
            
            try:
                with open(file_path, "rb") as f:
                    supabase.storage.from_(bucket_name).upload(
                        path=storage_path,
                        file=f,
                        file_options={"content-type": content_type, "upsert": "true"}
                    )
                print(" ✅")
                success_count += 1
            except Exception as e:
                print(f" ❌ Failed: {e}")
                fail_count += 1

    print("-" * 50)
    print(f"Upload complete!")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Source directory not found: {SOURCE_DIR}")
    else:
        upload_directory(SOURCE_DIR, BUCKET_NAME)
