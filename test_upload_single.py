"""
Test uploading a single image to Supabase to verify the service role key works
"""
import os
from dotenv import load_dotenv
from db import supabase, upload_image_to_storage, create_yolo_file
from pathlib import Path

load_dotenv()

print("=" * 60)
print("Testing Supabase Upload with Service Role Key")
print("=" * 60)

# Check if we have images to test with
test_image_path = None
dataset_path = Path("Healthy Pechay.v1i.yolov9/train/images")

if dataset_path.exists():
    # Find first image
    image_files = list(dataset_path.glob("*.jpg"))[:1]
    if image_files:
        test_image_path = str(image_files[0])
        print(f"\nFound test image: {os.path.basename(test_image_path)}")
    else:
        print("\n[WARN] No images found in dataset folder")
else:
    print(f"\n[WARN] Dataset folder not found: {dataset_path}")

if test_image_path:
    try:
        print("\n1. Testing Storage Upload...")
        image_url = upload_image_to_storage(test_image_path, "petchay-images")
        if image_url:
            print(f"   [OK] Image uploaded successfully!")
            print(f"   URL: {image_url[:80]}...")
        else:
            print(f"   [FAIL] Upload failed - returned None")
            
    except Exception as e:
        print(f"   [ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n2. Testing Database Insert (yolo_files)...")
        test_filename = f"test_healthy_{os.path.basename(test_image_path)}"
        result = create_yolo_file(
            filename=test_filename,
            file_type="image",
            dataset_type="Healthy",
            url=image_url or test_image_path,
            treatment=None
        )
        print(f"   [OK] Entry created successfully!")
        print(f"   Filename: {test_filename}")
        
    except Exception as e:
        print(f"   [ERROR] Database insert failed: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n[Skipping upload test - no test image available]")

# Test direct Supabase query
print("\n3. Testing Direct Supabase Query...")
try:
    result = supabase.table("yolo_files").select("count", count="exact").limit(1).execute()
    print(f"   [OK] Query successful")
    if hasattr(result, 'count'):
        print(f"   Total yolo_files: {result.count}")
except Exception as e:
    print(f"   [ERROR] Query failed: {e}")

print("\n" + "=" * 60)
print("If uploads fail, check:")
print("1. Service role key is correct (should start with 'eyJ' for JWT)")
print("2. Key has 'service_role' permissions in Supabase")
print("3. Storage bucket 'petchay-images' exists")
print("4. RLS policies allow service role to insert")
print("=" * 60)

