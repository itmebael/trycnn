# 📸 Scan Leaf Upload Flow - Complete Process

## Overview
When you upload an image in the **"📸 Scan Leaf"** page on the dashboard, here's exactly what happens:

---

## 🔄 Step-by-Step Process

### **Step 1: User Uploads Image**
- **Location**: Dashboard → "📸 Scan Leaf" page (`/dashboard?page=upload`)
- **Form**: User selects an image file and clicks "📸 Scan Petchay Leaf"
- **File Field**: `leafImage` (multipart/form-data)

### **Step 2: File Validation & Save** 
**Code**: `app.py` → `/dashboard` route (POST handler, line ~2594)

```python
# 1. Check if file exists
file = request.files.get("leafImage")

# 2. Validate file type
if not allowed_file(file.filename):
    upload_status = "File is not an image."

# 3. Save to local storage
filename = secure_filename(file.filename)
save_path = os.path.join(UPLOAD_FOLDER, filename)  # Usually: uploads/filename.jpg
file.save(save_path)
```

**What Happens**:
- ✅ File is validated (must be image: .jpg, .jpeg, .png, .gif, .webp)
- ✅ Filename is sanitized (removes dangerous characters)
- ✅ File is saved to `uploads/` folder locally

---

### **Step 3: Detection Process Starts**
**Code**: `app.py` → `detect_leaf_condition(image_path)` (line ~1411)

```python
detection_result = detect_leaf_condition(save_path)
```

**What Happens Inside `detect_leaf_condition()`**:

#### **3.1. Log Upload to Database**
```python
create_file_upload_log(
    filename=filename,
    file_path=image_path,
    file_size=file_size,
    upload_source="prediction",
    user_id=session.get("user_id")
)
```
- ✅ Logs upload to `file_uploads` table
- ✅ Tracks file size, source, user

#### **3.2. Upload to Supabase Storage** 🆕
```python
storage_url = upload_image_to_storage(image_path, STORAGE_BUCKET_NAME)
# Uploads to: petchay-images bucket
# Returns: https://...supabase.co/storage/v1/object/public/petchay-images/filename.jpg
```
- ✅ Image is uploaded to Supabase Storage (`petchay-images` bucket)
- ✅ Gets a public URL for the image
- ✅ Image is now accessible from anywhere

#### **3.3. Image Validation (Pechay Gate)**
```python
gate = pechay_color_gate(image_path)
if not gate.get("ok", True):
    # Reject if:
    # - Not green enough
    # - Round shape (pechay leaves are elongated)
    # - Digital/drawn image
    # - Contains non-pechay objects (people, cars, fruits, etc.)
    return {"condition": "Not Pechay", ...}
```
- ✅ Validates image is actually a pechay leaf
- ✅ Rejects digital images, round images, non-pechay objects
- ✅ Uses AI + YOLO for validation

#### **3.4. Detection Methods (Priority Order)**

**🔴 PRIMARY: Face Recognition Style Matching**
```python
face_recognition_match = face_recognition_style_match(image_path, cnn_predictor)
# Uses petchay_dataset embeddings
# Compares with existing healthy/diseased pechay images
# Returns match if similarity > 70%
```
- ✅ Extracts embedding from uploaded image
- ✅ Compares with `petchay_dataset` table embeddings
- ✅ Finds closest match using cosine similarity
- ✅ Returns condition, disease_name, treatment if match found

**🟡 SECONDARY: YOLO Files Database Comparison**
```python
comparison_result = compare_with_yolo_files_database(image_path)
# Uses yolo_files table
# Compares label, dataset_type, label_confidence
```
- ✅ Compares with `yolo_files` database entries
- ✅ Uses label matching and confidence scores
- ✅ Returns condition, disease_name, treatment

**🟢 FALLBACK: AI Models (CNN/YOLO)**
```python
# Only used if database matching fails
cnn_result = cnn_predictor.predict_image(image_path)
yolo_result = yolo_predict_image(image_path)
```
- ✅ Uses trained CNN model
- ✅ Uses YOLO model (if available)
- ✅ Provides AI-based prediction

#### **3.5. Generate Recommendations**
```python
recommendations = get_recommendations(
    condition, 
    confidence, 
    probabilities, 
    image_features,
    disease_name,
    treatment
)
```
- ✅ Generates care tips based on condition
- ✅ Provides treatment advice if diseased
- ✅ Includes urgency level

---

### **Step 4: Save Detection Result to Database**
**Code**: `app.py` → `/dashboard` route (line ~2607)

```python
create_detection_result(
    filename=filename,
    condition=detection_result["condition"],  # "Healthy" or "Diseased"
    confidence=detection_result.get("confidence", 0),  # 0-100%
    image_path=detection_result["image_path"],  # Local path or storage URL
    recommendations=detection_result.get("recommendations", {}),
    all_probabilities=detection_result.get("all_probabilities"),
    user_id=user_id,  # Links to user account
    timestamp=detection_result.get("timestamp"),
    disease_name=detection_result.get("disease_name"),  # e.g., "Leaf Spot"
    treatment=detection_result.get("treatment")  # Treatment advice
)
```

**What Gets Saved**:
- ✅ Detection result → `detection_results` table
- ✅ Condition (Healthy/Diseased)
- ✅ Confidence percentage
- ✅ Disease name (if diseased)
- ✅ Treatment information (if available)
- ✅ Recommendations
- ✅ Image path (local or storage URL)
- ✅ User ID (for user-specific results)
- ✅ Timestamp

---

### **Step 5: Display Result**
**Code**: `templates/dashboard.html` (line ~662)

**What User Sees**:
```html
🔍 Detection Result
├── Status: Healthy / Diseased
├── Disease: [Disease Name] (if diseased)
├── Treatment: [Treatment Info] (if available)
├── Confidence: 85%
├── Image Preview
└── Recommendations
    ├── Title
    ├── Tips (list)
    └── Action
```

**Result Card Shows**:
- ✅ Condition status (color-coded: green for healthy, red for diseased)
- ✅ Disease name (if detected)
- ✅ Treatment information (if available)
- ✅ Confidence percentage
- ✅ Image preview
- ✅ Care recommendations

---

## 📊 Data Flow Summary

```
User Uploads Image
    ↓
[1] File Saved Locally → uploads/filename.jpg
    ↓
[2] Upload Logged → file_uploads table
    ↓
[3] Uploaded to Storage → petchay-images bucket (Supabase)
    ↓
[4] Image Validation → pechay_color_gate()
    ↓
[5] Detection Process:
    ├─ Face Recognition Match (petchay_dataset)
    ├─ YOLO Files Database (yolo_files)
    └─ AI Models (CNN/YOLO) [fallback]
    ↓
[6] Result Saved → detection_results table
    ↓
[7] Display Result → Dashboard page
```

---

## 🗄️ Database Tables Updated

1. **`file_uploads`**
   - Logs every file upload
   - Tracks file size, source, user

2. **`detection_results`**
   - Stores detection result
   - Condition, confidence, disease_name, treatment
   - Links to user account

3. **Supabase Storage (`petchay-images` bucket)**
   - Stores image file
   - Provides public URL
   - Enables image access from anywhere

---

## 🔍 Detection Priority Order

1. **Face Recognition Style** (Primary)
   - Uses `petchay_dataset` embeddings
   - Cosine similarity matching
   - Returns: condition, disease_name, treatment

2. **YOLO Files Database** (Secondary)
   - Uses `yolo_files` table
   - Label/dataset_type matching
   - Returns: condition, disease_name, treatment

3. **AI Models** (Fallback)
   - CNN model prediction
   - YOLO model prediction
   - Returns: condition, confidence

---

## 📝 Key Differences: Scan Leaf vs Create Dataset

| Feature | Scan Leaf | Create Dataset |
|---------|-----------|----------------|
| **Purpose** | Detect condition | Build training dataset |
| **Storage** | `petchay-images/` | `petchay-images/dataset/{folder}/` |
| **Database** | `detection_results` | `yolo_files` + `petchay_dataset` |
| **Validation** | Yes (pechay gate) | Yes (pechay gate) |
| **Embeddings** | Generated for matching | Generated and saved |
| **Result** | Display detection | Save for training |

---

## 🎯 What Happens to the Image?

1. **Local Storage**: Saved to `uploads/filename.jpg`
2. **Supabase Storage**: Uploaded to `petchay-images` bucket
3. **Database**: Metadata saved to `detection_results` table
4. **Display**: Shown on dashboard with detection result

---

## ✅ Summary

When you upload an image in **Scan Leaf**:
- ✅ Image is validated (must be pechay leaf)
- ✅ Image is uploaded to Supabase Storage
- ✅ Detection runs using face recognition + database + AI
- ✅ Result is saved to database
- ✅ Result is displayed on the page
- ✅ Image is accessible via public URL

The system prioritizes **database matching** (face recognition style) over AI models, ensuring accurate detection based on your training dataset! 🎉

