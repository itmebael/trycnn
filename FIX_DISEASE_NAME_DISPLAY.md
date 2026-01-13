# Fix: Disease Name Not Displaying

## Problem
The `disease_name` field is not displaying in the detection results even when condition is "Diseased".

## Root Cause
The `disease_name` might not be extracted correctly from `yolo_files` table when:
1. `dataset_type` is "Diseased" (generic) instead of specific disease name
2. `label` is "Diseased" (generic) instead of specific disease name
3. Disease name exists but extraction logic misses it

## Solution Applied

### 1. Enhanced Disease Name Extraction
Updated `compare_with_yolo_files_database()` to:
- Check `dataset_type` first (if not "Healthy" or "Diseased")
- Check `label` second (if not "Healthy" or "Diseased")
- Check all diseased matches for disease names
- Add debug logging to track extraction

### 2. Multiple Extraction Points
Disease name is now extracted at:
1. Verified matches extraction
2. Non-verified matches extraction
3. Final check before returning result

### 3. Debug Logging
Added debug output to track:
- When disease_name is extracted
- When disease_name is missing
- What values are in dataset_type and label

## How to Verify

### Check Database
```sql
-- Check if yolo_files have disease names
SELECT filename, dataset_type, label, treatment 
FROM yolo_files 
WHERE dataset_type != 'Healthy' 
  AND dataset_type IS NOT NULL
LIMIT 10;
```

### Expected Values
For `disease_name` to display, entries should have:
- `dataset_type`: "Leaf Spot", "Downy Mildew", "Mosaic Virus", etc. (NOT "Diseased")
- OR `label`: "Leaf Spot", "Downy Mildew", etc. (NOT "Diseased")

### Example Correct Entry
```sql
INSERT INTO yolo_files (
    filename,
    dataset_type,  -- Should be specific: "Leaf Spot" NOT "Diseased"
    label,         -- Should be specific: "Leaf Spot" NOT "Diseased"
    treatment
) VALUES (
    'leaf_spot_001.jpg',
    'Leaf Spot',        -- ✅ Specific disease name
    'Leaf Spot',        -- ✅ Specific disease name
    'Apply fungicide...'
);
```

### Example Incorrect Entry (Won't Show Disease Name)
```sql
INSERT INTO yolo_files (
    filename,
    dataset_type,  -- ❌ Generic "Diseased"
    label          -- ❌ Generic "Diseased"
) VALUES (
    'diseased_001.jpg',
    'Diseased',    -- ❌ Too generic
    'Diseased'     -- ❌ Too generic
);
```

## Testing

After uploading an image:
1. Check console output for `[DEBUG]` messages
2. Check if `disease_name` appears in detection result
3. Verify in dashboard that disease name displays

## If Still Not Displaying

1. **Check database entries**: Ensure `dataset_type` or `label` contains specific disease names
2. **Check console logs**: Look for `[DEBUG]` or `[WARN]` messages
3. **Verify detection result**: Check if `disease_name` is in the returned dict
4. **Check template**: Ensure `{% if detection_result.disease_name %}` condition is met

The fix ensures disease_name is extracted from `dataset_type` or `label` fields when condition is "Diseased".

