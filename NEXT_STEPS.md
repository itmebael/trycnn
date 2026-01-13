# Next Steps: Complete Dataset Creation Setup

## ✅ What's Done

1. ✅ Database schema created with label fields
2. ✅ SQL scripts prepared (create_tables.sql, add_label_column.sql, etc.)
3. ✅ Treatment field added to dataset creation workflow
4. ✅ Image validation (round shape, color) implemented

## 📋 What's Next

### Step 1: Run Database Setup Scripts

**Option A: Fresh Database (Recommended)**
```sql
-- Run this in Supabase SQL Editor or psql
\i create_tables.sql
```

**Option B: Existing Database**
```sql
-- If tables already exist, run:
\i add_label_column.sql
\i fix_match_function.sql
```

**Verify tables were created:**
```sql
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('yolo_files', 'petchay_dataset');
```

---

### Step 2: Update Backend Code to Use Label Fields

Update `db.py` to include label fields when creating entries:

**File: `db.py`**

Update `create_yolo_file()` function:
```python
def create_yolo_file(
    filename: str,
    file_type: str = "image",
    dataset_type: str = None,
    url: str = None,
    treatment: Optional[str] = None,
    # Add new label parameters:
    label: Optional[str] = None,
    label_confidence: Optional[float] = None,
    image_region: Optional[str] = None,
    bounding_box: Optional[dict] = None,
    annotation_notes: Optional[str] = None,
    quality_score: Optional[float] = None,
    is_verified: bool = False
) -> Dict[str, Any]:
    data_entry = {
        "filename": filename,
        "file_type": file_type,
        "dataset_type": dataset_type,
        "url": url or filename,
        "treatment": treatment,
        # Add label fields:
        "label": label or dataset_type,  # Use dataset_type as fallback
        "label_confidence": label_confidence,
        "image_region": image_region,
        "bounding_box": bounding_box,
        "annotation_notes": annotation_notes,
        "quality_score": quality_score,
        "is_verified": is_verified,
        "uploaded_at": datetime.now().isoformat()
    }
    # ... rest of function
```

Update `save_dataset_entry()` function similarly.

---

### Step 3: Update Frontend to Collect Label Information

**File: `templates/dataset_manager.html`**

Add fields for:
- Image Region selection (leaf, stem, whole_plant, etc.)
- Annotation Notes textarea
- Quality Score (can be auto-calculated or manual)

Add to the form:
```html
<!-- After treatment field -->
<div class="workflow-step">
    <h4><span class="step-number">📸</span> Image Details</h4>
    
    <label>Image Region:</label>
    <select name="image_region">
        <option value="">-- Select region --</option>
        <option value="leaf">Leaf</option>
        <option value="stem">Stem</option>
        <option value="whole_plant">Whole Plant</option>
        <option value="multiple_leaves">Multiple Leaves</option>
        <option value="close_up">Close Up</option>
    </select>
    
    <label>Annotation Notes:</label>
    <textarea name="annotation_notes" rows="3" 
              placeholder="Additional notes about the image..."></textarea>
</div>
```

---

### Step 4: Update Upload Endpoints

**File: `app.py`**

Update `upload_image_immediate()` to capture label fields:
```python
# Get form data
image_region = request.form.get("image_region", "").strip()
annotation_notes = request.form.get("annotation_notes", "").strip()

# When creating yolo_file:
create_yolo_file(
    filename=unique_filename,
    file_type="image",
    dataset_type=folder_label,
    url=image_url,
    treatment=treatment if condition == "Diseased" and treatment else None,
    # Add label fields:
    label=disease_name if condition == "Diseased" else "Healthy",
    image_region=image_region if image_region else None,
    annotation_notes=annotation_notes if annotation_notes else None,
    is_verified=False  # Can be verified later
)
```

---

### Step 5: Add Image Quality Calculation (Optional)

Create a function to calculate quality score:

**File: `app.py`**

```python
def calculate_image_quality(image_path: str) -> float:
    """Calculate image quality score (0.0 to 1.0)"""
    try:
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        
        # Calculate brightness (0-1)
        brightness = np.mean(arr) / 255.0
        
        # Calculate contrast (standard deviation)
        contrast = np.std(arr) / 255.0
        
        # Calculate sharpness (edge detection)
        from scipy import ndimage
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        sharpness = np.std(edges) / 255.0
        
        # Combine scores (weighted average)
        quality = (brightness * 0.3 + contrast * 0.4 + sharpness * 0.3)
        
        return float(np.clip(quality, 0.0, 1.0))
    except Exception as e:
        print(f"Error calculating quality: {e}")
        return 0.5  # Default medium quality
```

Use it when uploading:
```python
quality_score = calculate_image_quality(save_path)
```

---

### Step 6: Test the Complete Workflow

1. **Start Flask server:**
   ```bash
   python app.py
   ```

2. **Test Create Dataset:**
   - Go to "Create Dataset"
   - Select "Diseased" → "Leaf Spot"
   - Enter treatment
   - Select image region
   - Add annotation notes
   - Upload images

3. **Verify in Database:**
   ```sql
   SELECT filename, label, image_region, annotation_notes, quality_score 
   FROM yolo_files 
   ORDER BY uploaded_at DESC 
   LIMIT 5;
   ```

---

### Step 7: Add Label Verification UI (Optional)

Create a page to verify labels:

**File: `templates/verify_labels.html`** (new file)

- List unverified images
- Allow users to verify/correct labels
- Update `is_verified` flag

---

## 🎯 Priority Order

1. **HIGH PRIORITY:**
   - ✅ Run database setup scripts
   - ✅ Update backend to save label fields
   - ✅ Test basic upload workflow

2. **MEDIUM PRIORITY:**
   - Add image region selection to frontend
   - Add annotation notes field
   - Calculate quality scores automatically

3. **LOW PRIORITY:**
   - Add label verification UI
   - Add bounding box editor
   - Add bulk verification tools

---

## 🔍 Quick Checklist

- [ ] Database tables created with label fields
- [ ] Backend code updated to save label fields
- [ ] Frontend updated to collect label information
- [ ] Upload endpoints updated
- [ ] Test upload workflow
- [ ] Verify data in database
- [ ] Test detection with labeled images

---

## 📝 Notes

- Label fields are **optional** - system works without them
- Start with basic fields (label, image_region) first
- Add advanced features (quality_score, bounding_box) later
- Quality score can be auto-calculated or manual

---

## 🆘 Troubleshooting

**If upload fails:**
- Check database connection
- Verify table columns exist
- Check Flask logs for errors

**If labels not saving:**
- Verify backend code includes label fields
- Check form field names match backend
- Verify database columns exist

**If detection not accurate:**
- Ensure images are properly labeled
- Verify embeddings are being generated
- Check quality scores are reasonable

---

## 🚀 Ready to Start?

1. Run: `create_tables.sql` or `add_label_column.sql`
2. Update: `db.py` and `app.py` 
3. Test: Upload an image and verify in database
4. Iterate: Add more features as needed

Good luck! 🎉
