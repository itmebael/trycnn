-- ============================================================
-- DATABASE SCHEMA FOR CREATE DATASET FUNCTIONALITY
-- Pechay Detection System - Dataset Management Tables
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector"; -- For embeddings (pgvector)

-- ============================================================
-- TABLE 1: yolo_files
-- Purpose: Tracks images in the YOLO training dataset
-- Used during scanning to check if an image is "known"
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
    annotation_notes TEXT NULL, -- Additional notes about the image (e.g., "close-up of infected area", "multiple spots visible")
    quality_score FLOAT NULL, -- Image quality score (0.0 to 1.0) - brightness, focus, clarity
    is_verified BOOLEAN NULL DEFAULT FALSE, -- Whether the label has been manually verified
    verified_by UUID NULL, -- User ID who verified the label (optional)
    verified_at TIMESTAMP WITH TIME ZONE NULL, -- When the label was verified
    uploaded_at TIMESTAMP WITHOUT TIME ZONE NULL DEFAULT NOW(),
    CONSTRAINT yolo_files_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- Indexes for faster lookups
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

-- Comments
COMMENT ON TABLE public.yolo_files IS 
    'Stores metadata about images in the YOLO training dataset. Used during scanning to check if an image is known.';

COMMENT ON COLUMN public.yolo_files.id IS 'Primary key, auto-generated';
COMMENT ON COLUMN public.yolo_files.filename IS 'Unique filename of the image file';
COMMENT ON COLUMN public.yolo_files.file_type IS 'Type of file (e.g., "image")';
COMMENT ON COLUMN public.yolo_files.dataset_type IS 'Dataset category: Healthy, Diseased, or specific disease name (e.g., "Leaf Spot", "Downy Mildew")';
COMMENT ON COLUMN public.yolo_files.url IS 'URL or path to the image file (Supabase Storage URL or local path)';
COMMENT ON COLUMN public.yolo_files.treatment IS 'Treatment recommendations or information for the disease/condition. Only populated when condition is Diseased.';
COMMENT ON COLUMN public.yolo_files.label IS 'Primary label for the image: Healthy, Diseased, or specific disease name. Used for accurate detection.';
COMMENT ON COLUMN public.yolo_files.label_confidence IS 'Confidence score of the label (0.0 to 1.0). Higher values indicate more reliable labels.';
COMMENT ON COLUMN public.yolo_files.image_region IS 'Region of pechay shown in image: leaf, stem, whole_plant, multiple_leaves, close_up. Helps with detection accuracy.';
COMMENT ON COLUMN public.yolo_files.bounding_box IS 'Bounding box coordinates in normalized format (0-1): {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.7}. Used for YOLO training.';
COMMENT ON COLUMN public.yolo_files.annotation_notes IS 'Additional notes about the image (e.g., "close-up of infected area", "multiple spots visible"). Helps with detection context.';
COMMENT ON COLUMN public.yolo_files.quality_score IS 'Image quality score (0.0 to 1.0) based on brightness, focus, clarity. Higher quality images improve detection accuracy.';
COMMENT ON COLUMN public.yolo_files.is_verified IS 'Whether the label has been manually verified by a user. Verified labels are more reliable for training.';
COMMENT ON COLUMN public.yolo_files.verified_by IS 'User ID who verified the label (optional).';
COMMENT ON COLUMN public.yolo_files.verified_at IS 'Timestamp when the label was verified.';
COMMENT ON COLUMN public.yolo_files.uploaded_at IS 'Timestamp when the file was uploaded';

-- ============================================================
-- TABLE 2: petchay_dataset
-- Purpose: Stores dataset entries with embeddings for Hybrid Detection
-- Supports similarity matching using vector embeddings
-- ============================================================

-- Create petchay_dataset table if it doesn't exist
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

-- Add updated_at column if table exists but column doesn't (for existing tables)
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE public.petchay_dataset 
        ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW();
    END IF;
END $$;

-- Indexes
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

-- Comments
COMMENT ON TABLE public.petchay_dataset IS 
    'Stores dataset entries with embeddings for Hybrid Detection (YOLO + CNN embeddings). Used for similarity matching during scanning.';

COMMENT ON COLUMN public.petchay_dataset.id IS 'Primary key, UUID';
COMMENT ON COLUMN public.petchay_dataset.filename IS 'Filename of the image';
COMMENT ON COLUMN public.petchay_dataset.condition IS 'Condition: Healthy or Diseased';
COMMENT ON COLUMN public.petchay_dataset.disease_name IS 'Specific disease name if condition is Diseased (e.g., Leaf Spot, Downy Mildew)';
COMMENT ON COLUMN public.petchay_dataset.image_url IS 'URL to the image file (Supabase Storage or local path)';
COMMENT ON COLUMN public.petchay_dataset.embedding IS 'CNN feature embeddings (512-dimensional vector) for similarity matching';
COMMENT ON COLUMN public.petchay_dataset.label IS 'Primary label for the image: Healthy, Diseased, or specific disease name. Used for accurate detection.';
COMMENT ON COLUMN public.petchay_dataset.label_confidence IS 'Confidence score of the label (0.0 to 1.0). Higher values indicate more reliable labels.';
COMMENT ON COLUMN public.petchay_dataset.image_region IS 'Region of pechay shown in image: leaf, stem, whole_plant, multiple_leaves, close_up. Helps with detection accuracy.';
COMMENT ON COLUMN public.petchay_dataset.bounding_box IS 'Bounding box coordinates in normalized format (0-1): {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.7}. Used for YOLO training.';
COMMENT ON COLUMN public.petchay_dataset.annotation_notes IS 'Additional notes about the image (e.g., "early stage infection", "severe damage"). Helps with detection context.';
COMMENT ON COLUMN public.petchay_dataset.quality_score IS 'Image quality score (0.0 to 1.0) based on brightness, focus, clarity. Higher quality images improve detection accuracy.';
COMMENT ON COLUMN public.petchay_dataset.is_verified IS 'Whether the label has been manually verified by a user. Verified labels are more reliable for training.';
COMMENT ON COLUMN public.petchay_dataset.verified_by IS 'User ID who verified the label (optional).';
COMMENT ON COLUMN public.petchay_dataset.verified_at IS 'Timestamp when the label was verified.';
COMMENT ON COLUMN public.petchay_dataset.user_id IS 'User who uploaded this dataset entry (optional)';
COMMENT ON COLUMN public.petchay_dataset.created_at IS 'Timestamp when entry was created';
COMMENT ON COLUMN public.petchay_dataset.updated_at IS 'Timestamp when entry was last updated';

-- ============================================================
-- TABLE 3: users (if not exists)
-- Purpose: User management for tracking who created datasets
-- ============================================================

CREATE TABLE IF NOT EXISTS public.users (
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL, -- Hashed password
    created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
    CONSTRAINT users_pkey PRIMARY KEY (id),
    CONSTRAINT users_email_key UNIQUE (email),
    CONSTRAINT users_username_key UNIQUE (username)
) TABLESPACE pg_default;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_users_username 
    ON public.users(username) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_users_email 
    ON public.users(email) TABLESPACE pg_default;

-- ============================================================
-- FUNCTION: match_petchay_embeddings
-- Purpose: Find similar images using vector similarity search
-- Used in Hybrid Detection for similarity matching
-- ============================================================

-- Drop function if it exists (to handle return type changes)
-- Drop all possible variations of the function signature
DROP FUNCTION IF EXISTS match_petchay_embeddings(VECTOR, FLOAT, INT);
DROP FUNCTION IF EXISTS match_petchay_embeddings(VECTOR, DOUBLE PRECISION, INTEGER);
DROP FUNCTION IF EXISTS match_petchay_embeddings(vector, double precision, integer);
DROP FUNCTION IF EXISTS match_petchay_embeddings(vector(512), float, int);
DROP FUNCTION IF EXISTS match_petchay_embeddings(vector(512), double precision, integer);
-- Drop with CASCADE to remove any dependencies
DROP FUNCTION IF EXISTS match_petchay_embeddings CASCADE;

-- Create function with correct return type
CREATE FUNCTION match_petchay_embeddings (
    query_embedding VECTOR(512),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    filename TEXT,
    condition TEXT,
    disease_name TEXT,
    image_url TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        pd.id,
        pd.filename,
        pd.condition,
        pd.disease_name,
        pd.image_url,
        1 - (pd.embedding <=> query_embedding) AS similarity
    FROM public.petchay_dataset pd
    WHERE pd.embedding IS NOT NULL
        AND 1 - (pd.embedding <=> query_embedding) > match_threshold
    ORDER BY pd.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_petchay_embeddings IS 
    'Finds similar pechay images using vector similarity search on embeddings. Returns top N matches above threshold.';

-- ============================================================
-- TRIGGER: Update updated_at timestamp
-- ============================================================

-- First, ensure updated_at column exists in petchay_dataset
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE public.petchay_dataset 
        ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW();
    END IF;
END $$;

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

-- Apply trigger to users
DROP TRIGGER IF EXISTS update_users_updated_at ON public.users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- SAMPLE QUERIES FOR REFERENCE
-- ============================================================

-- Query 1: Get all dataset entries
-- SELECT * FROM public.petchay_dataset ORDER BY created_at DESC;

-- Query 2: Get all yolo_files
-- SELECT * FROM public.yolo_files ORDER BY uploaded_at DESC;

-- Query 3: Get dataset entries by condition
-- SELECT * FROM public.petchay_dataset WHERE condition = 'Diseased';

-- Query 4: Get dataset entries by disease name
-- SELECT * FROM public.petchay_dataset WHERE disease_name = 'Leaf Spot';

-- Query 5: Get yolo_files with treatment information
-- SELECT filename, dataset_type, treatment, uploaded_at 
-- FROM public.yolo_files 
-- WHERE treatment IS NOT NULL 
-- ORDER BY uploaded_at DESC;

-- Query 6: Find similar images (example)
-- SELECT * FROM match_petchay_embeddings(
--     '[0.1, 0.2, ...]'::vector(512),  -- Your query embedding
--     0.7,  -- Similarity threshold
--     5     -- Number of results
-- );

-- Query 7: Count dataset entries by condition
-- SELECT condition, COUNT(*) as count 
-- FROM public.petchay_dataset 
-- GROUP BY condition;

-- Query 8: Count dataset entries by disease
-- SELECT disease_name, COUNT(*) as count 
-- FROM public.petchay_dataset 
-- WHERE condition = 'Diseased'
-- GROUP BY disease_name
-- ORDER BY count DESC;

-- ============================================================
-- END OF SCHEMA
-- ============================================================

