# Upload Healthy Pechay Dataset

This script uploads all images from the `Healthy Pechay.v1i.yolov9` dataset to Supabase for use in the pechay detection system.

## What It Does

1. **Reads all images** from `Healthy Pechay.v1i.yolov9/train/images` (1335 images)
2. **Uploads images** to Supabase Storage (`petchay-images` bucket)
3. **Generates embeddings** using the CNN model for similarity matching
4. **Adds to yolo_files table** with:
   - `condition`: "Healthy"
   - `dataset_type`: "Healthy"
   - `label`: "Healthy"
   - `label_confidence`: 1.0
   - `is_verified`: True
5. **Adds to petchay_dataset table** with:
   - `condition`: "Healthy"
   - `embedding`: 512-dim feature vector
   - `image_url`: Supabase storage URL

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install tqdm
   ```

2. **Ensure Supabase is configured**:
   - Set `SUPABASE_URL` in `.env`
   - Set `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_ANON_KEY` in `.env`

3. **Ensure CNN model exists**:
   - Model file: `pechay_cnn_model_20251212_184656.pth`
   - If not available, embeddings will be skipped but images will still be uploaded

## Usage

```bash
python upload_healthy_dataset.py
```

## What Happens

1. The script scans the dataset folder for all `.jpg`, `.jpeg`, and `.png` files
2. For each image:
   - Uploads to Supabase Storage
   - Generates embedding using CNN model
   - Adds entry to `yolo_files` table
   - Adds entry to `petchay_dataset` table with embedding
3. Shows progress bar and statistics

## Output

The script will display:
- Total images found
- Progress bar
- Success/failure counts for each table
- Final summary

## After Upload

Once uploaded, the system will:
- ✅ Use these images for healthy pechay detection
- ✅ Compare uploaded images with these embeddings
- ✅ Use yolo_files entries for YOLO model training
- ✅ Provide accurate "Healthy" detection results

## Notes

- The script processes images sequentially to avoid memory issues
- If an image fails to upload, it continues with the next one
- Embeddings are optional - if the model isn't available, images are still uploaded
- The script can be interrupted (Ctrl+C) and resumed later (duplicates will be skipped)

## Troubleshooting

**Error: "No images found"**
- Check that the dataset path is correct: `Healthy Pechay.v1i.yolov9/train/images`

**Error: "Model file not found"**
- Ensure `pechay_cnn_model_20251212_184656.pth` exists in the project root
- Images will still be uploaded without embeddings

**Error: "Supabase connection failed"**
- Check your `.env` file has correct Supabase credentials
- Ensure Supabase is accessible

**Slow upload**
- This is normal for 1335 images
- Each image needs to be uploaded and processed
- Estimated time: 30-60 minutes depending on connection speed

