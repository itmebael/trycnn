# YOLO Files Database as Primary Detection Model

## Overview

The Scan Leaf feature now uses the **`yolo_files` table as the PRIMARY detection model**. AI models (CNN/YOLO) are only used as **support/validation** when the database doesn't have enough matches.

## Detection Flow

### Step 1: Image Validation
```
Upload Image → Validate (pechay_color_gate)
    ├─ Reject non-pechay objects
    ├─ Reject digital images
    ├─ Reject round shapes
    └─ Check green color
```

### Step 2: PRIMARY MODEL - yolo_files Database
```
Compare with yolo_files table
    ├─ Get all entries (up to 500)
    ├─ Group by condition (Healthy vs Diseased)
    ├─ Prioritize verified entries (is_verified=True)
    ├─ Use label_confidence scores
    └─ Count matches
    ↓
If match found (confidence > 50%)
    ├─ Return result immediately ✅
    ├─ Include disease_name
    ├─ Include treatment
    └─ AI models used only for validation/boosting
```

### Step 3: SUPPORT - AI Models (Only if no database match)
```
If no database match:
    ├─ Use YOLO model (support)
    ├─ Use CNN model (support)
    └─ Use hybrid detection (fallback)
```

## How It Works

### PRIMARY: yolo_files Database Detection

1. **Query Database**
   ```python
   all_yolo_files = get_all_yolo_files(limit=500)
   ```

2. **Group by Condition**
   - Healthy: `dataset_type="Healthy"` OR `label="Healthy"`
   - Diseased: `dataset_type="Diseased"` OR `label="Diseased"` OR specific disease name

3. **Prioritize Verified Entries**
   - First check `is_verified=True` entries
   - Use average `label_confidence` from verified matches

4. **Calculate Confidence**
   - Based on match ratio (healthy_count vs diseased_count)
   - Weighted by `label_confidence` scores
   - Boosted if AI models agree

5. **Return Result**
   - Condition: Healthy or Diseased
   - Disease Name: From `dataset_type` or `label`
   - Treatment: From `treatment` field
   - Confidence: Based on database matches

### SUPPORT: AI Models (Validation Only)

AI models are **only used** to:
1. **Validate** database results (if they agree, boost confidence)
2. **Support** when database has no matches
3. **Fallback** if everything else fails

**AI Support Logic:**
- If database says "Healthy" and AI embedding similarity > 70% → Boost confidence
- If database says "Diseased" and YOLO agrees → Boost confidence
- AI never overrides database results, only supports them

## Database Schema Used

The system uses these fields from `yolo_files`:

```sql
- dataset_type: "Healthy", "Diseased", or disease name
- label: "Healthy", "Diseased", or disease name
- label_confidence: 0.0 to 1.0 (confidence score)
- is_verified: true/false (prioritized)
- quality_score: 0.0 to 1.0 (image quality)
- treatment: Treatment recommendations
```

## Detection Priority

1. **PRIMARY**: `yolo_files` database (verified entries first)
2. **SUPPORT**: AI embeddings (CNN) - only for validation
3. **SUPPORT**: YOLO model - only for validation
4. **FALLBACK**: Hybrid detection - only if all else fails

## Example Detection Flow

### Scenario 1: Database Has Match
```
Upload Image
    ↓
Query yolo_files (500 entries)
    ↓
Found: 50 Healthy, 10 Diseased
    ↓
Result: Healthy (confidence: 85%)
    ↓
AI Support: Embedding similarity 82% → Boost to 88%
    ↓
Return: Healthy, 88% confidence ✅
```

### Scenario 2: Database Has No Match
```
Upload Image
    ↓
Query yolo_files (500 entries)
    ↓
Found: 0 matches
    ↓
Use AI Support: YOLO detects "Healthy" (75%)
    ↓
Return: Healthy, 75% confidence (from AI support)
```

### Scenario 3: Database Has Weak Match
```
Upload Image
    ↓
Query yolo_files (500 entries)
    ↓
Found: 5 Healthy, 3 Diseased (weak match)
    ↓
Result: Healthy (confidence: 55%)
    ↓
AI Support: YOLO agrees → Boost to 65%
    ↓
Return: Healthy, 65% confidence ✅
```

## Benefits

✅ **Database-First**: Uses your curated dataset as primary source  
✅ **Accurate**: Relies on verified labels with confidence scores  
✅ **Fast**: Database queries are faster than AI inference  
✅ **Reliable**: Verified entries (`is_verified=True`) prioritized  
✅ **AI Support**: AI models validate and boost confidence  
✅ **Treatment Info**: Includes treatment from database  

## Configuration

The system automatically:
- Queries up to 500 entries from `yolo_files`
- Prioritizes verified entries
- Uses `label_confidence` for weighting
- Falls back to AI only if database has no matches

## Adding Data to yolo_files

To improve detection accuracy, add more entries to `yolo_files`:

```sql
INSERT INTO yolo_files (
    filename,
    dataset_type,  -- "Healthy" or "Diseased" or disease name
    label,         -- "Healthy" or "Diseased" or disease name
    label_confidence,  -- 0.0 to 1.0
    is_verified,   -- true for verified entries
    treatment      -- Treatment recommendations
) VALUES (...);
```

The more verified entries you have, the more accurate the detection! 🎯

