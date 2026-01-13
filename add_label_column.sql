-- ============================================================
-- ADD LABEL COLUMN: Quick fix for missing label column
-- Creates tables if they don't exist, then adds label columns
-- ============================================================

-- Ensure required extensions exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- First, create yolo_files table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.yolo_files (
    id BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NULL DEFAULT 'image',
    dataset_type TEXT NULL,
    url TEXT NULL,
    treatment TEXT NULL,
    uploaded_at TIMESTAMP WITHOUT TIME ZONE NULL DEFAULT NOW(),
    CONSTRAINT yolo_files_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- Create basic indexes for yolo_files if they don't exist
CREATE INDEX IF NOT EXISTS idx_yolo_files_filename 
    ON public.yolo_files(filename) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_dataset_type 
    ON public.yolo_files(dataset_type) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_uploaded_at 
    ON public.yolo_files(uploaded_at DESC) TABLESPACE pg_default;

-- Create petchay_dataset table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.petchay_dataset (
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    condition TEXT NOT NULL,
    disease_name TEXT NULL,
    image_url TEXT NOT NULL,
    embedding VECTOR(512),
    user_id UUID NULL,
    created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
    CONSTRAINT petchay_dataset_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- Create basic indexes for petchay_dataset if they don't exist
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

-- Add label column to yolo_files if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'label'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN label TEXT NULL;
        RAISE NOTICE 'Added label column to yolo_files';
    ELSE
        RAISE NOTICE 'label column already exists in yolo_files';
    END IF;
END $$;

-- Add label column to petchay_dataset if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'label'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN label TEXT NULL;
        -- Set label from condition for existing rows
        UPDATE public.petchay_dataset SET label = condition WHERE label IS NULL;
        RAISE NOTICE 'Added label column to petchay_dataset';
    ELSE
        RAISE NOTICE 'label column already exists in petchay_dataset';
    END IF;
END $$;

-- Add other label-related columns to yolo_files
DO $$ 
BEGIN
    -- label_confidence
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'label_confidence'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN label_confidence FLOAT NULL;
        RAISE NOTICE 'Added label_confidence to yolo_files';
    END IF;
    
    -- image_region
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'image_region'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN image_region TEXT NULL;
        RAISE NOTICE 'Added image_region to yolo_files';
    END IF;
    
    -- bounding_box
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'bounding_box'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN bounding_box JSONB NULL;
        RAISE NOTICE 'Added bounding_box to yolo_files';
    END IF;
    
    -- annotation_notes
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'annotation_notes'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN annotation_notes TEXT NULL;
        RAISE NOTICE 'Added annotation_notes to yolo_files';
    END IF;
    
    -- quality_score
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'quality_score'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN quality_score FLOAT NULL;
        RAISE NOTICE 'Added quality_score to yolo_files';
    END IF;
    
    -- is_verified
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'is_verified'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN is_verified BOOLEAN NULL DEFAULT FALSE;
        RAISE NOTICE 'Added is_verified to yolo_files';
    END IF;
    
    -- verified_by
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'verified_by'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN verified_by UUID NULL;
        RAISE NOTICE 'Added verified_by to yolo_files';
    END IF;
    
    -- verified_at
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'yolo_files' AND column_name = 'verified_at'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN verified_at TIMESTAMP WITH TIME ZONE NULL;
        RAISE NOTICE 'Added verified_at to yolo_files';
    END IF;
END $$;

-- Add other label-related columns to petchay_dataset
DO $$ 
BEGIN
    -- label_confidence
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'label_confidence'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN label_confidence FLOAT NULL;
        RAISE NOTICE 'Added label_confidence to petchay_dataset';
    END IF;
    
    -- image_region
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'image_region'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN image_region TEXT NULL;
        RAISE NOTICE 'Added image_region to petchay_dataset';
    END IF;
    
    -- bounding_box
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'bounding_box'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN bounding_box JSONB NULL;
        RAISE NOTICE 'Added bounding_box to petchay_dataset';
    END IF;
    
    -- annotation_notes
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'annotation_notes'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN annotation_notes TEXT NULL;
        RAISE NOTICE 'Added annotation_notes to petchay_dataset';
    END IF;
    
    -- quality_score
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'quality_score'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN quality_score FLOAT NULL;
        RAISE NOTICE 'Added quality_score to petchay_dataset';
    END IF;
    
    -- is_verified
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'is_verified'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN is_verified BOOLEAN NULL DEFAULT FALSE;
        RAISE NOTICE 'Added is_verified to petchay_dataset';
    END IF;
    
    -- verified_by
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'verified_by'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN verified_by UUID NULL;
        RAISE NOTICE 'Added verified_by to petchay_dataset';
    END IF;
    
    -- verified_at
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'petchay_dataset' AND column_name = 'verified_at'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN verified_at TIMESTAMP WITH TIME ZONE NULL;
        RAISE NOTICE 'Added verified_at to petchay_dataset';
    END IF;
END $$;

-- Create indexes for label columns
CREATE INDEX IF NOT EXISTS idx_yolo_files_label 
    ON public.yolo_files(label) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_label 
    ON public.petchay_dataset(label) TABLESPACE pg_default;

-- Verify label column exists
SELECT 
    'yolo_files' as table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'yolo_files'
AND column_name = 'label';

SELECT 
    'petchay_dataset' as table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'petchay_dataset'
AND column_name = 'label';

-- Add updated_at column to petchay_dataset if it doesn't exist
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
        
        -- Update existing rows
        UPDATE public.petchay_dataset 
        SET updated_at = created_at 
        WHERE updated_at IS NULL;
        
        RAISE NOTICE 'Added updated_at column to petchay_dataset';
    END IF;
END $$;

-- Create trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for petchay_dataset
DROP TRIGGER IF EXISTS update_petchay_dataset_updated_at ON public.petchay_dataset;
CREATE TRIGGER update_petchay_dataset_updated_at
    BEFORE UPDATE ON public.petchay_dataset
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Success message
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created and label columns added!';
    RAISE NOTICE '  - yolo_files table created/updated';
    RAISE NOTICE '  - petchay_dataset table created/updated';
    RAISE NOTICE '  - label column added to both tables';
    RAISE NOTICE '  - All label-related columns added';
    RAISE NOTICE '========================================';
END $$;

