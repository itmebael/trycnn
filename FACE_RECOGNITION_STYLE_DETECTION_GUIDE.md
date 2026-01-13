# Face Recognition Style Detection for Pechay

## Overview

The system now uses **face recognition-style matching** using `petchay_dataset` embeddings. This works exactly like face recognition - it compares the uploaded image's embedding with all embeddings in the database to find the closest match.

## How It Works

### Step 1: Extract Embedding
```
Upload Image → CNN Model → Extract 512-dim embedding vector
```

### Step 2: Compare with Database (Face Recognition Style)
```
For each entry in petchay_dataset:
    ├─ Get embedding vector
    ├─ Calculate cosine similarity
    └─ Track best match (highest similarity)
```

### Step 3: Find Best Match
```
If similarity > 70% (threshold):
    ├─ Return match ✅
    ├─ Get condition (Healthy/Diseased)
    ├─ Get disease_name
    ├─ Get treatment from yolo_files
    └─ Return result
```

### Step 4: Fallback to yolo_files
```
If no embedding match:
    └─ Use yolo_files database matching
```

## Detection Priority

1. **PRIMARY**: Face Recognition Style (`petchay_dataset` embeddings)
   - Extracts embedding from uploaded image
   - Compares with all embeddings in `petchay_dataset`
   - Finds closest match using cosine similarity
   - Returns match if similarity > 70%

2. **SECONDARY**: `yolo_files` Database Matching
   - Uses `dataset_type`, `label`, `label_confidence`
   - Groups by condition (Healthy vs Diseased)
   - Returns match if confidence > 50%

3. **SUPPORT**: AI Models (CNN/YOLO)
   - Only used if database matching fails
   - Provides fallback detection

## Face Recognition Matching Details

### Similarity Calculation
```python
cosine_similarity = dot_product / (norm1 * norm2)
```

### Threshold
- **Similarity Threshold**: 0.7 (70%)
- **Confidence**: similarity * 100
- **Match Required**: similarity >= 0.7

### Example Flow

```
Upload Image
    ↓
Extract Embedding (512-dim vector)
    ↓
Compare with 1335 embeddings in petchay_dataset
    ↓
Found: Best match similarity = 0.85 (85%)
    ↓
Match Entry: condition="Healthy", disease_name=None
    ↓
Check yolo_files for treatment
    ↓
Return: Healthy, 85% confidence ✅
```

## Database Requirements

### petchay_dataset Table
- Must have `embedding` column (vector(512))
- Embeddings must be generated using CNN model
- More embeddings = better matching accuracy

### yolo_files Table
- Used for treatment information
- Matched by filename if embedding match found
- Provides `treatment` field

## Benefits

✅ **Face Recognition Accuracy**: Uses same technique as face recognition  
✅ **Fast Matching**: Vector similarity search is very fast  
✅ **Scalable**: Can handle thousands of embeddings  
✅ **Accurate**: Cosine similarity finds closest visual match  
✅ **Treatment Info**: Gets treatment from yolo_files  

## How to Add More Data

### Upload Healthy Pechay Dataset
```bash
python upload_healthy_dataset.py
```

This will:
1. Upload images to Supabase Storage
2. Generate embeddings using CNN model
3. Add entries to `petchay_dataset` with embeddings
4. Add entries to `yolo_files` for treatment lookup

### Add Diseased Pechay Images
Use the "Create Dataset" workflow:
1. Select "Diseased"
2. Enter disease name (e.g., "Leaf Spot")
3. Upload images
4. System generates embeddings automatically
5. Images added to both `petchay_dataset` and `yolo_files`

## Example Detection

### Scenario 1: Healthy Pechay Match
```
Upload Image
    ↓
Face Recognition: Similarity = 0.88 (88%)
    ↓
Match: condition="Healthy"
    ↓
Result: Healthy, 88% confidence ✅
```

### Scenario 2: Diseased Pechay Match
```
Upload Image
    ↓
Face Recognition: Similarity = 0.82 (82%)
    ↓
Match: condition="Diseased", disease_name="Leaf Spot"
    ↓
Get treatment from yolo_files
    ↓
Result: Diseased, Leaf Spot, Treatment, 82% confidence ✅
```

### Scenario 3: No Match
```
Upload Image
    ↓
Face Recognition: Best similarity = 0.55 (55%) < 70% threshold
    ↓
Fallback to yolo_files database matching
    ↓
Result: Based on yolo_files matches
```

## Technical Details

### Embedding Extraction
- Uses CNN model (`PechayPredictor`)
- Extracts 512-dim feature vector
- Normalized for cosine similarity

### Similarity Search
- Compares with all embeddings in `petchay_dataset`
- Uses cosine similarity (dot product / norms)
- Finds highest similarity match

### Performance
- Fast: Vector operations are optimized
- Scalable: Can handle 10,000+ embeddings
- Accurate: Cosine similarity is proven for image matching

## Configuration

The face recognition matching uses:
- **Similarity Threshold**: 0.7 (70%)
- **Embedding Dimension**: 512
- **Match Count**: 1 (best match only)

You can adjust the threshold in `face_recognition_style_detection.py`:
```python
similarity_threshold = 0.7  # Change this to adjust sensitivity
```

## Troubleshooting

**No matches found?**
- Check if `petchay_dataset` has entries with embeddings
- Verify embeddings are generated correctly
- Lower similarity threshold if needed

**Low similarity scores?**
- Add more similar images to database
- Ensure images are similar quality/lighting
- Check if CNN model is working correctly

**Treatment not showing?**
- Check if `yolo_files` has matching filename
- Verify `treatment` field is populated
- Check filename matching logic

The system now works like face recognition - it finds the closest matching pechay image in your database! 🎯

