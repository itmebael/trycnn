"""
Create Supabase Storage Bucket for Dataset Images
"""
from db import supabase

print("=" * 60)
print("Creating Supabase Storage Bucket for Dataset")
print("=" * 60)

bucket_name = "petchay-images"  # Use same bucket as scan images

try:
    # Check if bucket exists
    print(f"\n1. Checking if bucket '{bucket_name}' exists...")
    try:
        buckets = supabase.storage.list_buckets()
        existing_buckets = [b.name for b in buckets]
        
        if bucket_name in existing_buckets:
            print(f"   [OK] Bucket '{bucket_name}' already exists!")
            print(f"   [INFO] Using existing bucket for dataset images")
        else:
            print(f"   [INFO] Bucket '{bucket_name}' not found. Creating...")
            
            # Create bucket
            result = supabase.storage.create_bucket(
                bucket_name,
                options={
                    "public": True,  # Make it public so images can be accessed
                    "file_size_limit": 52428800,  # 50MB limit
                    "allowed_mime_types": ["image/jpeg", "image/png", "image/jpg", "image/gif"]
                }
            )
            print(f"   [OK] Bucket '{bucket_name}' created successfully!")
            
    except Exception as e:
        print(f"   [ERROR] Failed to create bucket: {e}")
        print(f"\n   Manual steps:")
        print(f"   1. Go to Supabase Dashboard > Storage")
        print(f"   2. Click 'New bucket'")
        print(f"   3. Name: {bucket_name}")
        print(f"   4. Make it PUBLIC")
        print(f"   5. Click 'Create bucket'")
        
except Exception as e:
    print(f"\n[ERROR] Storage operation failed: {e}")
    print(f"\nThis might be because:")
    print(f"1. Service role key doesn't have storage permissions")
    print(f"2. Storage API is not enabled")
    print(f"\nManual fix:")
    print(f"1. Go to Supabase Dashboard > Storage")
    print(f"2. Create bucket manually: {bucket_name}")
    print(f"3. Make it PUBLIC")
    print(f"4. Set file size limit to 50MB")

print("\n" + "=" * 60)
print("Note: Dataset images will be uploaded to 'petchay-images' bucket")
print("This is the same bucket used for scan images")
print("=" * 60)

