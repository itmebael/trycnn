# Fix: Dataset Images Not Saving to Storage

## Problem
When uploading images in the "Create / Update Dataset" workflow, images are not being saved to Supabase Storage.

## Root Cause
1. **Wrong Bucket Name**: Code was trying to upload to `petchay-dataset` bucket which doesn't exist
2. **Silent Failures**: Upload errors were not being logged properly
3. **Missing Error Handling**: No feedback when storage upload fails

## Solution Applied

### 1. Fixed Bucket Name
Changed all occurrences from:
```python
bucket_name=os.getenv("DATASET_BUCKET_NAME", "petchay-dataset")  # ❌ Doesn't exist
```

To:
```python
bucket_name=os.getenv("DATASET_BUCKET_NAME", "petchay-images")  # ✅ Exists
```

### 2. Added Logging
Added detailed logging to track upload process:
```python
print(f"[UPLOAD] Uploading to bucket: {dataset_bucket}")
print(f"[UPLOAD] ✅ Image uploaded to storage: {image_url}")
print(f"[UPLOAD] ⚠️ Storage upload failed, using local path")
```

### 3. Improved Storage Path Organization
Updated `upload_image_to_storage()` to organize images:
- Dataset images: `dataset/{folder}/{filename}` (e.g., `dataset/Healthy/image.jpg`)
- Scan images: `{filename}` (e.g., `image.jpg`)

### 4. Enhanced Error Handling
Added detailed error logging with traceback to identify issues.

## Files Updated

1. **`app.py`**:
   - `/upload_image_immediate` endpoint
   - `/upload_training_data` endpoint
   - `/upload_dataset_workflow` endpoint
   - All now use `petchay-images` bucket

2. **`db.py`**:
   - `upload_image_to_storage()` function
   - Added folder structure for dataset images
   - Enhanced error logging

## How to Verify

### Check Console Output
When uploading images, you should see:
```
[UPLOAD] Uploading to bucket: petchay-images
[STORAGE] Uploading image.jpg to petchay-images/dataset/Healthy/image.jpg
[STORAGE] ✅ Upload successful: https://...supabase.co/storage/v1/object/public/petchay-images/dataset/Healthy/image.jpg
[UPLOAD] ✅ Image uploaded to storage: https://...
```

### Check Supabase Storage
1. Go to Supabase Dashboard > Storage
2. Open `petchay-images` bucket
3. You should see:
   - `dataset/Healthy/` folder
   - `dataset/Diseased/` folder (or disease names)
   - Images inside these folders

### Check Database
```sql
-- Check yolo_files entries
SELECT filename, url FROM yolo_files 
WHERE url LIKE '%storage%' 
ORDER BY uploaded_at DESC 
LIMIT 10;

-- Check petchay_dataset entries
SELECT filename, image_url FROM petchay_dataset 
WHERE image_url LIKE '%storage%' 
ORDER BY created_at DESC 
LIMIT 10;
```

## If Still Not Working

### Check 1: Bucket Exists
```bash
python create_dataset_storage_bucket.py
```

### Check 2: Service Role Key
Ensure `.env` has:
```
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

### Check 3: Storage Permissions
In Supabase Dashboard:
1. Go to Storage > Policies
2. Ensure service role can upload files
3. Check bucket is PUBLIC

### Check 4: Console Errors
Look for error messages in:
- Browser console (F12)
- Flask server console
- Check for `[STORAGE] ❌` messages

## Expected Behavior

After fix:
1. ✅ Images upload to `petchay-images` bucket
2. ✅ Images organized in `dataset/{folder}/` structure
3. ✅ URLs stored in database (`yolo_files.url`, `petchay_dataset.image_url`)
4. ✅ Images accessible via public URLs
5. ✅ Detailed logging shows upload status

The fix ensures all dataset images are properly uploaded to Supabase Storage! 🎉

