-- ============================================================
-- CREATE TABLES: yolo_files and petchay_dataset
-- Creates tables with all fields including label fields
-- Safe to run multiple times (IF NOT EXISTS)
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================================
-- TABLE 1: yolo_files
-- Purpose: Tracks images in the YOLO training dataset
-- ============================================================

CREATE TABLE IF NOT EXISTS public.yolo_files (
    id BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NULL DEFAULT 'image',
    dataset_type TEXT NULL, -- 'Healthy', 'Diseased', or specific disease name
    url TEXT NULL, -- URL or path to the image file
    treatment TEXT NULL, -- Treatment recommendations for the disease/condition
    
    -- Label and Annotation Fields for Accurate Detection
    label TEXT NULL, -- Primary label: 'Healthy', 'Diseased', or specific disease name
    label_confidence FLOAT NULL, -- Confidence score of the label (0.0 to 1.0)
    image_region TEXT NULL, -- Region of pechay shown: 'leaf', 'stem', 'whole_plant', 'multiple_leaves'
    bounding_box JSONB NULL, -- Bounding box coordinates: {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.7} (normalized 0-1)
    annotation_notes TEXT NULL, -- Additional notes about the image
    quality_score FLOAT NULL, -- Image quality score (0.0 to 1.0) - brightness, focus, clarity
    is_verified BOOLEAN NULL DEFAULT FALSE, -- Whether the label has been manually verified
    verified_by UUID NULL, -- User ID who verified the label (optional)
    verified_at TIMESTAMP WITH TIME ZONE NULL, -- When the label was verified
    
    uploaded_at TIMESTAMP WITHOUT TIME ZONE NULL DEFAULT NOW(),
    CONSTRAINT yolo_files_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- Indexes for yolo_files
CREATE INDEX IF NOT EXISTS idx_yolo_files_filename 
    ON public.yolo_files(filename) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_dataset_type 
    ON public.yolo_files(dataset_type) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_uploaded_at 
    ON public.yolo_files(uploaded_at DESC) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_label 
    ON public.yolo_files(label) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_image_region 
    ON public.yolo_files(image_region) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_is_verified 
    ON public.yolo_files(is_verified) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_quality_score 
    ON public.yolo_files(quality_score DESC) TABLESPACE pg_default;

-- ============================================================
-- TABLE 2: petchay_dataset
-- Purpose: Stores dataset entries with embeddings for Hybrid Detection
-- ============================================================

CREATE TABLE IF NOT EXISTS public.petchay_dataset (
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    condition TEXT NOT NULL, -- 'Healthy' or 'Diseased'
    disease_name TEXT NULL, -- Specific disease name (e.g., 'Leaf Spot', 'Downy Mildew', 'Mosaic Virus', 'Yellowing')
    image_url TEXT NOT NULL, -- URL to the image (Supabase Storage or local)
    embedding VECTOR(512), -- CNN feature embeddings (ResNet18 typically produces 512-dim vectors)
    
    -- Label and Annotation Fields for Accurate Detection
    label TEXT NULL, -- Primary label: same as condition or disease_name
    label_confidence FLOAT NULL, -- Confidence score of the label (0.0 to 1.0)
    image_region TEXT NULL, -- Region of pechay shown: 'leaf', 'stem', 'whole_plant', 'multiple_leaves', 'close_up'
    bounding_box JSONB NULL, -- Bounding box coordinates: {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.7} (normalized 0-1)
    annotation_notes TEXT NULL, -- Additional notes about the image (e.g., "early stage infection", "severe damage")
    quality_score FLOAT NULL, -- Image quality score (0.0 to 1.0) - brightness, focus, clarity
    is_verified BOOLEAN NULL DEFAULT FALSE, -- Whether the label has been manually verified
    verified_by UUID NULL, -- User ID who verified the label (optional)
    verified_at TIMESTAMP WITH TIME ZONE NULL, -- When the label was verified
    
    user_id UUID NULL, -- Reference to users table (optional)
    created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
    CONSTRAINT petchay_dataset_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- Indexes for petchay_dataset
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_filename 
    ON public.petchay_dataset(filename) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_condition 
    ON public.petchay_dataset(condition) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_disease_name 
    ON public.petchay_dataset(disease_name) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_user_id 
    ON public.petchay_dataset(user_id) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_created_at 
    ON public.petchay_dataset(created_at DESC) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_label 
    ON public.petchay_dataset(label) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_image_region 
    ON public.petchay_dataset(image_region) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_is_verified 
    ON public.petchay_dataset(is_verified) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_quality_score 
    ON public.petchay_dataset(quality_score DESC) TABLESPACE pg_default;

-- Vector similarity index (for fast embedding searches)
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_embedding 
    ON public.petchay_dataset USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================
-- TRIGGER: Update updated_at timestamp
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to petchay_dataset
DROP TRIGGER IF EXISTS update_petchay_dataset_updated_at ON public.petchay_dataset;
CREATE TRIGGER update_petchay_dataset_updated_at
    BEFORE UPDATE ON public.petchay_dataset
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- VERIFY TABLES WERE CREATED
-- ============================================================

-- Check if tables exist
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns 
     WHERE table_schema = 'public' 
     AND table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public' 
AND table_name IN ('yolo_files', 'petchay_dataset')
ORDER BY table_name;

-- Show column names for yolo_files
SELECT 
    'yolo_files' as table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'yolo_files'
ORDER BY ordinal_position;

-- Show column names for petchay_dataset
SELECT 
    'petchay_dataset' as table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'petchay_dataset'
ORDER BY ordinal_position;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created successfully!';
    RAISE NOTICE '  - yolo_files';
    RAISE NOTICE '  - petchay_dataset';
    RAISE NOTICE 'All label fields included for accurate detection';
    RAISE NOTICE '========================================';
END $$;

