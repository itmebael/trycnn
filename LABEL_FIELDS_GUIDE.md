# Label Fields Guide for Accurate Detection

## Overview

New label and annotation fields have been added to improve detection accuracy. These fields store detailed information about images that helps the system make more accurate predictions.

---

## New Fields Added

### 1. **`label`** (TEXT)
- **Purpose**: Primary label for the image
- **Values**: `'Healthy'`, `'Diseased'`, or specific disease name (e.g., `'Leaf Spot'`, `'Downy Mildew'`)
- **Usage**: Used as the main classification label for training and detection

### 2. **`label_confidence`** (FLOAT)
- **Purpose**: Confidence score of the label
- **Range**: `0.0` to `1.0` (higher = more reliable)
- **Usage**: Filter high-quality labels for training, prioritize verified labels

### 3. **`image_region`** (TEXT)
- **Purpose**: Region of pechay shown in the image
- **Values**: 
  - `'leaf'` - Single leaf or leaf detail
  - `'stem'` - Stem portion
  - `'whole_plant'` - Entire pechay plant
  - `'multiple_leaves'` - Multiple leaves visible
  - `'close_up'` - Close-up of specific area
- **Usage**: Helps model understand context and improve detection accuracy

### 4. **`bounding_box`** (JSONB)
- **Purpose**: Bounding box coordinates for YOLO training
- **Format**: Normalized coordinates (0.0 to 1.0)
  ```json
  {
    "x": 0.1,      // Left edge (10% from left)
    "y": 0.2,      // Top edge (20% from top)
    "width": 0.8,  // Width (80% of image width)
    "height": 0.7  // Height (70% of image height)
  }
  ```
- **Usage**: Used for YOLO object detection training

### 5. **`annotation_notes`** (TEXT)
- **Purpose**: Additional notes about the image
- **Examples**:
  - `"close-up of infected area"`
  - `"multiple spots visible"`
  - `"early stage infection"`
  - `"severe damage"`
- **Usage**: Provides context for better understanding and detection

### 6. **`quality_score`** (FLOAT)
- **Purpose**: Image quality score
- **Range**: `0.0` to `1.0` (higher = better quality)
- **Factors**: Brightness, focus, clarity, contrast
- **Usage**: Filter low-quality images from training, prioritize high-quality samples

### 7. **`is_verified`** (BOOLEAN)
- **Purpose**: Whether label has been manually verified
- **Default**: `FALSE`
- **Usage**: Only use verified labels for training to improve accuracy

### 8. **`verified_by`** (UUID)
- **Purpose**: User ID who verified the label
- **Usage**: Track who verified labels for quality control

### 9. **`verified_at`** (TIMESTAMP)
- **Purpose**: When the label was verified
- **Usage**: Track verification history

---

## How to Use These Fields

### During Image Upload

When creating a dataset, you can now provide:

```python
create_yolo_file(
    filename="abc123_image.jpg",
    file_type="image",
    dataset_type="Leaf Spot",
    url="https://...",
    treatment="Apply fungicide...",
    # New label fields:
    label="Leaf Spot",
    label_confidence=0.95,
    image_region="leaf",
    bounding_box={"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.7},
    annotation_notes="Close-up showing multiple brown spots",
    quality_score=0.9,
    is_verified=True,
    verified_by=user_id
)
```

### Querying High-Quality Labels

```sql
-- Get verified labels only
SELECT * FROM petchay_dataset 
WHERE is_verified = TRUE 
ORDER BY label_confidence DESC;

-- Get high-quality images for training
SELECT * FROM petchay_dataset 
WHERE quality_score >= 0.8 
AND label_confidence >= 0.9
ORDER BY quality_score DESC;

-- Get images by region
SELECT * FROM petchay_dataset 
WHERE image_region = 'leaf' 
AND condition = 'Diseased';
```

### Using Bounding Boxes for YOLO Training

```sql
-- Get images with bounding boxes
SELECT filename, label, bounding_box 
FROM petchay_dataset 
WHERE bounding_box IS NOT NULL;

-- Example bounding_box value:
-- {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.7}
```

---

## Benefits for Detection Accuracy

### 1. **Better Training Data**
- Verified labels ensure model learns from correct examples
- Quality scores filter out poor images
- Image region helps model understand context

### 2. **Improved Detection**
- Label confidence helps prioritize reliable predictions
- Bounding boxes enable precise object detection
- Annotation notes provide context for edge cases

### 3. **Quality Control**
- Verification system tracks label accuracy
- Quality scores identify problematic images
- Region labels help categorize detection scenarios

---

## Example Workflow

### Step 1: Upload Image with Labels
```python
# Upload image
image_url = upload_image_to_storage("pechay_leaf.jpg")

# Create entry with labels
create_yolo_file(
    filename="pechay_leaf.jpg",
    dataset_type="Leaf Spot",
    url=image_url,
    label="Leaf Spot",
    label_confidence=0.95,
    image_region="leaf",
    bounding_box={"x": 0.05, "y": 0.1, "width": 0.9, "height": 0.8},
    annotation_notes="Early stage infection, small brown spots",
    quality_score=0.92
)
```

### Step 2: Verify Label (Optional)
```sql
-- Mark as verified
UPDATE petchay_dataset 
SET is_verified = TRUE,
    verified_by = 'user-uuid',
    verified_at = NOW()
WHERE filename = 'pechay_leaf.jpg';
```

### Step 3: Use for Training
```python
# Get verified, high-quality labels for training
verified_images = get_dataset_entries(
    is_verified=True,
    min_quality_score=0.8,
    min_confidence=0.9
)
```

---

## Migration

If you have existing tables, run the migration script:

```sql
-- Run this to add new columns
\i add_label_fields_migration.sql
```

This will:
1. Add all new columns to existing tables
2. Create indexes for performance
3. Set default values for existing rows
4. Verify columns were added successfully

---

## Best Practices

1. **Always set `label`** - This is the primary classification
2. **Use `image_region`** - Helps model understand context
3. **Set `quality_score`** - Filter low-quality images
4. **Verify important labels** - Set `is_verified = TRUE` for critical samples
5. **Add `annotation_notes`** - Provide context for edge cases
6. **Use `bounding_box`** - For precise object detection training

---

## SQL Examples

### Get all verified labels
```sql
SELECT filename, label, label_confidence, image_region
FROM petchay_dataset
WHERE is_verified = TRUE;
```

### Get high-quality training samples
```sql
SELECT * FROM petchay_dataset
WHERE quality_score >= 0.8
AND label_confidence >= 0.9
AND is_verified = TRUE
ORDER BY quality_score DESC, label_confidence DESC;
```

### Count labels by region
```sql
SELECT image_region, COUNT(*) as count
FROM petchay_dataset
GROUP BY image_region
ORDER BY count DESC;
```

### Get images with bounding boxes
```sql
SELECT filename, label, bounding_box
FROM petchay_dataset
WHERE bounding_box IS NOT NULL
AND bounding_box != 'null'::jsonb;
```

---

## Notes

- All new fields are **optional** (NULL allowed)
- Existing data will have NULL values for new fields
- You can populate them gradually as you verify images
- The system will work with or without these fields
- Fields improve accuracy when populated

