# Fix Supabase Storage Bucket Issue

## Problem
The upload script fails with: `Bucket not found`

## Solution

### Option 1: Create Bucket via Supabase Dashboard (Recommended)

1. **Go to Supabase Dashboard**
   - Visit: https://supabase.com/dashboard
   - Select your project

2. **Navigate to Storage**
   - Click "Storage" in the left sidebar
   - Click "New bucket"

3. **Create Bucket**
   - **Name**: `petchay-images`
   - **Public bucket**: ✅ Check this (so images can be accessed)
   - **File size limit**: 50 MB (or leave default)
   - Click "Create bucket"

4. **Verify**
   - You should see `petchay-images` in the bucket list
   - Status should show as "Public"

### Option 2: Create via SQL (Alternative)

Run this in Supabase SQL Editor:

```sql
-- Create storage bucket
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'petchay-images',
  'petchay-images',
  true,
  52428800,  -- 50MB
  ARRAY['image/jpeg', 'image/png', 'image/jpg', 'image/gif']
);
```

### Option 3: Use Python Script

Run:
```bash
python create_storage_bucket.py
```

## Verify It Works

After creating the bucket, test with:
```bash
python test_upload_single.py
```

You should see:
- ✅ Image uploaded successfully!
- ✅ Entry created successfully!

## Service Role Key

Your service role key is now set in `.env`:
```
SUPABASE_SERVICE_ROLE_KEY=sb_secret_JGU_P__Kcf0yc0nB2m3aPA_S0upaaji
```

**Note**: If the key format looks unusual (starts with `sb_secret_`), verify it's correct:
1. Go to Supabase Dashboard > Settings > API
2. Copy the "service_role" key (should start with `eyJ` for JWT format)
3. Update `.env` if needed

## After Fixing

Once the bucket is created, you can run:
```bash
python upload_healthy_dataset.py
```

This will upload all 1335 healthy pechay images to Supabase! 🎉

