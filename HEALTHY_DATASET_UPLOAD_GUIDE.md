# Healthy Pechay Dataset Upload Guide

## Overview

This guide explains how to upload the **Healthy Pechay.v1i.yolov9** dataset (1335 images) to Supabase for use in the pechay detection system.

## What Gets Uploaded

### 1. Images to Supabase Storage
- All 1335 healthy pechay images
- Stored in `petchay-images` bucket
- Used for serving images to the frontend

### 2. Metadata to `yolo_files` Table
Each image gets an entry with:
- `filename`: Unique filename
- `file_type`: "image"
- `dataset_type`: "Healthy"
- `url`: Supabase storage URL
- `label`: "Healthy"
- `label_confidence`: 1.0
- `image_region`: "leaf"
- `quality_score`: 0.9
- `is_verified`: True

### 3. Embeddings to `petchay_dataset` Table
Each image gets an entry with:
- `filename`: Unique filename
- `condition`: "Healthy"
- `image_url`: Supabase storage URL
- `embedding`: 512-dim feature vector (for similarity matching)
- `disease_name`: None (healthy, no disease)

## How to Run

### Step 1: Install Dependencies (Optional)
```bash
pip install tqdm
```
*Note: The script works without tqdm, but progress bar won't be shown*

### Step 2: Run the Upload Script
```bash
python upload_healthy_dataset.py
```

### Step 3: Wait for Completion
- The script processes all 1335 images
- Shows progress every 100 images
- Estimated time: 30-60 minutes (depending on connection)

## What Happens During Upload

1. **Scans Dataset Folder**
   - Finds all `.jpg`, `.jpeg`, `.png` files
   - Total: 1335 images

2. **For Each Image**:
   - Uploads to Supabase Storage
   - Generates embedding using CNN model
   - Adds to `yolo_files` table
   - Adds to `petchay_dataset` table with embedding

3. **Progress Updates**:
   - Shows progress every 100 images
   - Displays success/failure counts

4. **Final Summary**:
   - Total images processed
   - Success/failure counts for each table
   - Number of embeddings generated

## After Upload

Once uploaded, the system will:

### ✅ Use for Detection
- When users upload images, the system compares with these healthy pechay embeddings
- Uses cosine similarity to find closest matches
- Returns "Healthy" if similarity is high

### ✅ Use for YOLO Training
- The `yolo_files` entries are used for YOLO model training
- Helps improve detection accuracy

### ✅ Use for Comparison
- The `petchay_dataset` entries with embeddings enable:
  - Fast similarity search
  - Accurate healthy pechay detection
  - Confidence scoring based on similarity

## System Integration

The uploaded dataset integrates with:

1. **`compare_with_yolo_files_database()` function**
   - Compares uploaded images with `yolo_files` entries
   - Uses `dataset_type="Healthy"` to identify healthy pechay

2. **`hybrid_detection()` function**
   - Uses `petchay_dataset` embeddings for similarity matching
   - Finds closest healthy pechay matches

3. **Detection Results**
   - Shows "Healthy" condition when match is found
   - Displays confidence based on similarity score

## Troubleshooting

### Error: "Dataset path not found"
**Solution**: Check that `Healthy Pechay.v1i.yolov9/train/images` exists

### Error: "Model file not found"
**Solution**: 
- Ensure `pechay_cnn_model_20251212_184656.pth` exists
- Images will still be uploaded without embeddings

### Error: "Supabase connection failed"
**Solution**:
- Check `.env` file has correct Supabase credentials
- Ensure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set

### Slow Upload
**Normal**: Processing 1335 images takes time
- Each image needs to be uploaded
- Embeddings need to be generated
- Database entries need to be created

### Interrupted Upload
**Solution**: 
- Can restart the script
- Duplicate entries will be skipped (based on filename)

## Verification

After upload, verify in Supabase:

1. **Check `yolo_files` table**:
   ```sql
   SELECT COUNT(*) FROM yolo_files WHERE dataset_type = 'Healthy';
   ```
   Should return ~1335

2. **Check `petchay_dataset` table**:
   ```sql
   SELECT COUNT(*) FROM petchay_dataset WHERE condition = 'Healthy';
   ```
   Should return ~1335

3. **Check embeddings**:
   ```sql
   SELECT COUNT(*) FROM petchay_dataset 
   WHERE condition = 'Healthy' AND embedding IS NOT NULL;
   ```
   Should return ~1335 (if model was available)

## Next Steps

1. ✅ Dataset uploaded
2. ✅ System ready to detect healthy pechay
3. ✅ Users can upload images and get accurate "Healthy" detection
4. ✅ YOLO model can be trained with this dataset

The system is now ready to use the healthy pechay dataset for detection! 🎉

