# YOLO Files Database Comparison System

## Overview

The system now compares uploaded images with the `yolo_files` database table using **AI (embeddings)** and **YOLO** to detect if pechay is **Healthy** or **Diseased**.

---

## How It Works

### Step 1: User Uploads Image
```
User uploads image → System validates (checks for pechay, rejects non-pechay objects)
```

### Step 2: Compare with Database
```
System queries yolo_files table → Gets all entries (up to 200)
```

### Step 3: YOLO Detection
```
YOLO model detects objects in uploaded image
- If detects "healthy" → Healthy
- If detects "diseased" or disease name → Diseased
```

### Step 4: AI Embedding Comparison
```
CNN extracts features (512-dim embedding) from uploaded image
Compares with embeddings in petchay_dataset table
Finds most similar images using cosine similarity
```

### Step 5: Database Label Matching
```
Compares with yolo_files entries:
- Groups by condition (Healthy vs Diseased)
- Counts matches
- Uses label_confidence scores
```

### Step 6: Combine Results
```
Combines:
- YOLO detection (weighted 80%)
- AI embedding similarity (if >75% match)
- Database label matching (fallback)

Returns: condition (Healthy/Diseased), confidence, disease_name
```

---

## Database Schema

The `yolo_files` table structure:

```sql
CREATE TABLE public.yolo_files (
    id BIGINT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT DEFAULT 'image',
    dataset_type TEXT,              -- 'Healthy', 'Diseased', or disease name
    url TEXT,                       -- Image URL
    treatment TEXT,                 -- Treatment info
    label TEXT,                     -- Primary label
    label_confidence FLOAT,         -- Confidence (0.0-1.0)
    image_region TEXT,              -- 'leaf', 'stem', 'whole_plant', etc.
    bounding_box JSONB,             -- Bounding box coordinates
    annotation_notes TEXT,          -- Additional notes
    quality_score FLOAT,            -- Image quality (0.0-1.0)
    is_verified BOOLEAN,             -- Verified label
    verified_by UUID,               -- Who verified
    verified_at TIMESTAMP,          -- When verified
    uploaded_at TIMESTAMP DEFAULT NOW()
);
```

---

## Detection Flow

```
Upload Image
    ↓
Validate (pechay_color_gate)
    ├─ Reject non-pechay objects
    ├─ Reject digital images
    ├─ Reject round shapes
    └─ Check green color
    ↓
Compare with yolo_files Database
    ├─ YOLO Detection
    │   └─ Detect objects/condition
    ├─ AI Embedding Comparison
    │   └─ Find similar images (cosine similarity)
    └─ Database Label Matching
        └─ Count Healthy vs Diseased matches
    ↓
Combine Results
    ├─ If confidence > 70% → Return matched result
    └─ Else → Continue with YOLO/CNN prediction
    ↓
Return Result
    ├─ Condition: Healthy or Diseased
    ├─ Confidence: 0-100%
    ├─ Disease Name: (if Diseased)
    └─ Treatment: (from matched file)
```

---

## Key Functions

### `compare_with_yolo_files_database(image_path)`

**Purpose**: Compare uploaded image with yolo_files database

**Returns**:
```python
{
    "matched": True/False,
    "condition": "Healthy" or "Diseased",
    "disease_name": "Leaf Spot" or None,
    "confidence": 0.0-100.0,
    "matched_file": "filename.jpg",
    "yolo_detection": {...},
    "embedding_similarity": 0.0-1.0,
    "healthy_matches": 10,
    "diseased_matches": 5
}
```

**How it works**:
1. Gets all yolo_files entries (up to 200)
2. Runs YOLO detection on uploaded image
3. Extracts AI embedding from uploaded image
4. Compares embedding with petchay_dataset entries
5. Groups yolo_files by condition (Healthy/Diseased)
6. Combines all signals to determine condition

### `get_all_yolo_files(limit=500)`

**Purpose**: Get all entries from yolo_files table

**Returns**: List of yolo_file entries

### `get_yolo_files_by_condition(condition, limit=100)`

**Purpose**: Get yolo_files filtered by condition

**Parameters**:
- `condition`: "Healthy" or "Diseased" (optional)
- `limit`: Maximum number of entries

**Returns**: List of matching yolo_file entries

---

## Detection Accuracy

### Confidence Levels

- **High Confidence (>80%)**: 
  - YOLO detects condition + AI embedding matches (>75%)
  - Multiple database matches with high label_confidence

- **Medium Confidence (60-80%)**:
  - YOLO detection OR AI embedding match
  - Database label matching with moderate confidence

- **Low Confidence (<60%)**:
  - Falls back to standard YOLO/CNN prediction
  - No strong database matches

### Factors Affecting Accuracy

1. **Database Size**: More entries = better matching
2. **Label Quality**: Verified labels (`is_verified=True`) are more reliable
3. **Image Quality**: Higher `quality_score` = better matches
4. **Label Confidence**: Higher `label_confidence` = more reliable

---

## Example Usage

### When User Uploads Image:

```python
# 1. Image uploaded
image_path = "uploads/user_image.jpg"

# 2. System compares with database
result = compare_with_yolo_files_database(image_path)

# 3. If good match found
if result["matched"] and result["confidence"] > 70:
    condition = result["condition"]  # "Healthy" or "Diseased"
    confidence = result["confidence"]  # 85.5
    disease_name = result["disease_name"]  # "Leaf Spot" or None
    
    # Return result to user
    return {
        "condition": condition,
        "confidence": confidence,
        "disease_name": disease_name,
        "treatment": get_treatment_from_matched_file(result["matched_file"])
    }
```

---

## Database Query Examples

### Get All Healthy Images
```sql
SELECT * FROM yolo_files 
WHERE dataset_type = 'Healthy' 
   OR label = 'Healthy'
ORDER BY label_confidence DESC, quality_score DESC
LIMIT 50;
```

### Get All Diseased Images
```sql
SELECT * FROM yolo_files 
WHERE dataset_type != 'Healthy' 
  AND dataset_type IS NOT NULL
ORDER BY label_confidence DESC, quality_score DESC
LIMIT 50;
```

### Get Verified High-Quality Images
```sql
SELECT * FROM yolo_files 
WHERE is_verified = TRUE 
  AND quality_score >= 0.8
  AND label_confidence >= 0.9
ORDER BY uploaded_at DESC;
```

---

## Improving Detection Accuracy

### 1. Add More Training Data
- Upload more pechay images to yolo_files
- Label them correctly (Healthy/Diseased)
- Verify important labels (`is_verified=True`)

### 2. Improve Label Quality
- Set `label_confidence` accurately
- Verify labels manually
- Add `annotation_notes` for context

### 3. Use Quality Scores
- Set `quality_score` for each image
- Filter low-quality images from comparison
- Prioritize high-quality matches

### 4. Add Disease Names
- Specify exact disease names (Leaf Spot, Downy Mildew, etc.)
- This helps with more accurate matching

---

## Troubleshooting

**Problem**: Low confidence matches
**Solution**: 
- Add more images to yolo_files database
- Verify labels (`is_verified=True`)
- Improve image quality scores

**Problem**: Wrong condition detected
**Solution**:
- Check label accuracy in database
- Verify `dataset_type` and `label` fields
- Increase `label_confidence` for reliable entries

**Problem**: No matches found
**Solution**:
- Ensure yolo_files table has entries
- Check database connection
- Verify images are properly labeled

---

## Benefits

✅ **More Accurate**: Uses database of known images  
✅ **Faster**: Can return results without full YOLO/CNN prediction  
✅ **Reliable**: Uses verified labels from database  
✅ **Context-Aware**: Can provide treatment info from matched files  
✅ **Scalable**: Works with large databases efficiently  

---

## Next Steps

1. **Populate Database**: Upload pechay images to yolo_files
2. **Label Accurately**: Set correct labels and confidence scores
3. **Verify Labels**: Mark important entries as verified
4. **Test**: Upload images and verify detection accuracy
5. **Iterate**: Improve based on results

The system is now ready to compare uploaded images with your yolo_files database! 🎉

