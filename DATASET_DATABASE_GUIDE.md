# Database Schema Guide for Create Dataset Functionality

## Overview

The Create Dataset functionality uses **two main tables** to store dataset information:

1. **`yolo_files`** - Tracks images for YOLO training and scanning
2. **`petchay_dataset`** - Stores dataset entries with embeddings for Hybrid Detection

---

## Table 1: `yolo_files`

### Purpose
Tracks metadata about images in the YOLO training dataset. Used during scanning to check if an image is "known" (already in the dataset).

### Schema

```sql
CREATE TABLE public.yolo_files (
    id BIGINT PRIMARY KEY (auto-generated),
    filename TEXT NOT NULL,
    file_type TEXT DEFAULT 'image',
    dataset_type TEXT,  -- 'Healthy', 'Diseased', or disease name
    url TEXT,           -- Image URL/path
    treatment TEXT,     -- Treatment info (only for Diseased)
    uploaded_at TIMESTAMP DEFAULT NOW()
);
```

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGINT | Primary key, auto-generated |
| `filename` | TEXT | Unique filename (e.g., `abc123_image.jpg`) |
| `file_type` | TEXT | Type of file (usually `'image'`) |
| `dataset_type` | TEXT | Category: `'Healthy'`, `'Diseased'`, or specific disease name (e.g., `'Leaf Spot'`) |
| `url` | TEXT | URL to image (Supabase Storage URL or local path) |
| `treatment` | TEXT | Treatment recommendations (only populated when condition is Diseased) |
| `uploaded_at` | TIMESTAMP | When the file was uploaded |

### Example Data

```json
{
  "id": 1,
  "filename": "a1b2c3d4_pechay_leaf.jpg",
  "file_type": "image",
  "dataset_type": "Leaf Spot",
  "url": "https://supabase.co/storage/v1/object/public/petchay-dataset/Leaf%20Spot/a1b2c3d4_pechay_leaf.jpg",
  "treatment": "Apply copper-based fungicide. Remove infected leaves. Improve air circulation.",
  "uploaded_at": "2024-01-15 10:30:00"
}
```

### Usage

- **During Upload**: When user uploads images via "Create Dataset", entries are created here
- **During Scanning**: System checks this table to see if scanned image is already in dataset
- **Treatment Storage**: Treatment field stores user-entered treatment information for diseases

---

## Table 2: `petchay_dataset`

### Purpose
Stores dataset entries with **CNN embeddings** for Hybrid Detection (similarity matching). This enables the system to find similar images even if they weren't in the original YOLO training set.

### Schema

```sql
CREATE TABLE public.petchay_dataset (
    id UUID PRIMARY KEY (auto-generated),
    filename TEXT NOT NULL,
    condition TEXT NOT NULL,      -- 'Healthy' or 'Diseased'
    disease_name TEXT,             -- Specific disease name
    image_url TEXT NOT NULL,
    embedding VECTOR(512),         -- CNN feature embeddings
    user_id UUID,                  -- Optional: who uploaded it
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key, auto-generated UUID |
| `filename` | TEXT | Filename of the image |
| `condition` | TEXT | `'Healthy'` or `'Diseased'` |
| `disease_name` | TEXT | Specific disease name (e.g., `'Leaf Spot'`, `'Downy Mildew'`) |
| `image_url` | TEXT | URL to the image file |
| `embedding` | VECTOR(512) | 512-dimensional feature vector from CNN (ResNet18) |
| `user_id` | UUID | Optional: Reference to users table |
| `created_at` | TIMESTAMP | When entry was created |
| `updated_at` | TIMESTAMP | When entry was last updated |

### Example Data

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "a1b2c3d4_pechay_leaf.jpg",
  "condition": "Diseased",
  "disease_name": "Leaf Spot",
  "image_url": "https://supabase.co/storage/v1/object/public/petchay-dataset/Leaf%20Spot/a1b2c3d4_pechay_leaf.jpg",
  "embedding": [0.123, 0.456, ..., 0.789],  // 512 numbers
  "user_id": "user-uuid-here",
  "created_at": "2024-01-15 10:30:00",
  "updated_at": "2024-01-15 10:30:00"
}
```

### Usage

- **During Upload**: Embeddings are generated and saved here
- **During Scanning**: System uses similarity matching to find similar images
- **Hybrid Detection**: Combines YOLO detection with embedding similarity

---

## Workflow: How Data Flows

### 1. User Creates Dataset

```
User clicks "Create Dataset"
  ↓
Selects: "Diseased" → "Leaf Spot"
  ↓
Enters Treatment: "Apply fungicide..."
  ↓
Uploads Images
```

### 2. Backend Processing

For each uploaded image:

1. **Image Saved** → `uploads/dataset_custom/Leaf Spot/filename.jpg`
2. **Validation** → Check if image is actually pechay (color, shape)
3. **Embedding Generated** → CNN extracts 512-dim feature vector
4. **Upload to Storage** → Image uploaded to Supabase Storage
5. **Save to `yolo_files`**:
   ```python
   create_yolo_file(
       filename="abc123_image.jpg",
       file_type="image",
       dataset_type="Leaf Spot",
       url="https://...",
       treatment="Apply fungicide..."
   )
   ```
6. **Save to `petchay_dataset`**:
   ```python
   save_dataset_entry(
       filename="abc123_image.jpg",
       label="Diseased",
       image_url="https://...",
       embedding=[0.123, 0.456, ...],  # 512 numbers
       disease_name="Leaf Spot",
       user_id="user-uuid"
   )
   ```

### 3. During Scanning

When user scans a leaf:

1. **Check `yolo_files`** → Is this image already known?
2. **If not found** → Run YOLO detection
3. **Generate embedding** → Extract features from scanned image
4. **Query `petchay_dataset`** → Find similar images using `match_petchay_embeddings()`
5. **Return result** → Combine YOLO + similarity matching

---

## Key Functions

### `create_yolo_file()`
Creates entry in `yolo_files` table.

**Parameters:**
- `filename`: Image filename
- `file_type`: Usually `'image'`
- `dataset_type`: `'Healthy'`, `'Diseased'`, or disease name
- `url`: Image URL
- `treatment`: Treatment info (optional, only for Diseased)

### `save_dataset_entry()`
Creates entry in `petchay_dataset` table with embeddings.

**Parameters:**
- `filename`: Image filename
- `label`: `'Healthy'` or `'Diseased'`
- `image_url`: Image URL
- `embedding`: 512-dim feature vector
- `disease_name`: Disease name (optional)
- `user_id`: User ID (optional)

### `match_petchay_embeddings()`
Finds similar images using vector similarity.

**Parameters:**
- `query_embedding`: 512-dim vector to search for
- `match_threshold`: Minimum similarity (0.0-1.0)
- `match_count`: Number of results to return

**Returns:** List of similar images with similarity scores

---

## SQL Queries

### Get all dataset entries
```sql
SELECT * FROM public.petchay_dataset 
ORDER BY created_at DESC;
```

### Get entries by condition
```sql
SELECT * FROM public.petchay_dataset 
WHERE condition = 'Diseased';
```

### Get entries by disease
```sql
SELECT * FROM public.petchay_dataset 
WHERE disease_name = 'Leaf Spot';
```

### Get yolo_files with treatment
```sql
SELECT filename, dataset_type, treatment, uploaded_at 
FROM public.yolo_files 
WHERE treatment IS NOT NULL 
ORDER BY uploaded_at DESC;
```

### Count by condition
```sql
SELECT condition, COUNT(*) as count 
FROM public.petchay_dataset 
GROUP BY condition;
```

### Count by disease
```sql
SELECT disease_name, COUNT(*) as count 
FROM public.petchay_dataset 
WHERE condition = 'Diseased'
GROUP BY disease_name
ORDER BY count DESC;
```

### Find similar images
```sql
SELECT * FROM match_petchay_embeddings(
    '[0.1, 0.2, ...]'::vector(512),  -- Your query embedding
    0.7,  -- Similarity threshold
    5     -- Number of results
);
```

---

## File Structure

When images are uploaded, they are saved to:

```
uploads/
  └── dataset_custom/
      ├── Healthy/
      │   ├── abc123_image1.jpg
      │   └── def456_image2.jpg
      └── Leaf Spot/
          ├── ghi789_image3.jpg
          └── jkl012_image4.jpg
```

---

## Notes

1. **Treatment Field**: Only populated when `condition = 'Diseased'`
2. **Embeddings**: Generated using ResNet18 CNN (512 dimensions)
3. **Vector Search**: Uses pgvector extension for fast similarity matching
4. **Offline Mode**: System also saves to local JSON files (`yolo_files.json`, `custom_dataset.json`) as backup

---

## Setup Instructions

1. **Run the SQL file**:
   ```bash
   psql -U your_user -d your_database -f create_dataset_database.sql
   ```

2. **Or in Supabase Dashboard**:
   - Go to SQL Editor
   - Paste the contents of `create_dataset_database.sql`
   - Run it

3. **Verify tables exist**:
   ```sql
   SELECT table_name 
   FROM information_schema.tables 
   WHERE table_schema = 'public' 
   AND table_name IN ('yolo_files', 'petchay_dataset');
   ```

---

## Troubleshooting

**Problem**: `vector` extension not found
**Solution**: Install pgvector extension:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Problem**: Embeddings are NULL
**Solution**: Ensure CNN model is loaded and `extract_features()` is working

**Problem**: Treatment not saving
**Solution**: Check that `condition = 'Diseased'` and treatment field is not empty

