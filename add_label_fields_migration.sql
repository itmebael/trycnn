-- ============================================================
-- MIGRATION: Add Label Fields for Accurate Detection
-- Adds label, bounding box, annotation, and quality fields
-- Creates tables if they don't exist, then adds new columns
-- ============================================================

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

-- Ensure required extensions exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Add label fields to yolo_files table
DO $$ 
BEGIN
    -- Add label column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'label'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN label TEXT NULL;
        RAISE NOTICE 'Added label column to yolo_files';
    END IF;
    
    -- Add label_confidence column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'label_confidence'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN label_confidence FLOAT NULL;
        RAISE NOTICE 'Added label_confidence column to yolo_files';
    END IF;
    
    -- Add image_region column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'image_region'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN image_region TEXT NULL;
        RAISE NOTICE 'Added image_region column to yolo_files';
    END IF;
    
    -- Add bounding_box column (JSONB)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'bounding_box'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN bounding_box JSONB NULL;
        RAISE NOTICE 'Added bounding_box column to yolo_files';
    END IF;
    
    -- Add annotation_notes column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'annotation_notes'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN annotation_notes TEXT NULL;
        RAISE NOTICE 'Added annotation_notes column to yolo_files';
    END IF;
    
    -- Add quality_score column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'quality_score'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN quality_score FLOAT NULL;
        RAISE NOTICE 'Added quality_score column to yolo_files';
    END IF;
    
    -- Add is_verified column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'is_verified'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN is_verified BOOLEAN NULL DEFAULT FALSE;
        RAISE NOTICE 'Added is_verified column to yolo_files';
    END IF;
    
    -- Add verified_by column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'verified_by'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN verified_by UUID NULL;
        RAISE NOTICE 'Added verified_by column to yolo_files';
    END IF;
    
    -- Add verified_at column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'verified_at'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN verified_at TIMESTAMP WITH TIME ZONE NULL;
        RAISE NOTICE 'Added verified_at column to yolo_files';
    END IF;
END $$;

-- Add label fields to petchay_dataset table
DO $$ 
BEGIN
    -- Add label column
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
    END IF;
    
    -- Add label_confidence column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'label_confidence'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN label_confidence FLOAT NULL;
        RAISE NOTICE 'Added label_confidence column to petchay_dataset';
    END IF;
    
    -- Add image_region column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'image_region'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN image_region TEXT NULL;
        RAISE NOTICE 'Added image_region column to petchay_dataset';
    END IF;
    
    -- Add bounding_box column (JSONB)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'bounding_box'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN bounding_box JSONB NULL;
        RAISE NOTICE 'Added bounding_box column to petchay_dataset';
    END IF;
    
    -- Add annotation_notes column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'annotation_notes'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN annotation_notes TEXT NULL;
        RAISE NOTICE 'Added annotation_notes column to petchay_dataset';
    END IF;
    
    -- Add quality_score column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'quality_score'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN quality_score FLOAT NULL;
        RAISE NOTICE 'Added quality_score column to petchay_dataset';
    END IF;
    
    -- Add is_verified column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'is_verified'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN is_verified BOOLEAN NULL DEFAULT FALSE;
        RAISE NOTICE 'Added is_verified column to petchay_dataset';
    END IF;
    
    -- Add verified_by column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'verified_by'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN verified_by UUID NULL;
        RAISE NOTICE 'Added verified_by column to petchay_dataset';
    END IF;
    
    -- Add verified_at column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'petchay_dataset' 
        AND column_name = 'verified_at'
    ) THEN
        ALTER TABLE public.petchay_dataset ADD COLUMN verified_at TIMESTAMP WITH TIME ZONE NULL;
        RAISE NOTICE 'Added verified_at column to petchay_dataset';
    END IF;
END $$;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_yolo_files_label 
    ON public.yolo_files(label) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_image_region 
    ON public.yolo_files(image_region) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_is_verified 
    ON public.yolo_files(is_verified) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_quality_score 
    ON public.yolo_files(quality_score DESC) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_label 
    ON public.petchay_dataset(label) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_image_region 
    ON public.petchay_dataset(image_region) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_is_verified 
    ON public.petchay_dataset(is_verified) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_quality_score 
    ON public.petchay_dataset(quality_score DESC) TABLESPACE pg_default;

-- Verify columns were added
SELECT 
    'yolo_files' as table_name,
    column_name, 
    data_type, 
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'yolo_files'
AND column_name IN ('label', 'label_confidence', 'image_region', 'bounding_box', 'annotation_notes', 'quality_score', 'is_verified', 'verified_by', 'verified_at')
ORDER BY column_name;

SELECT 
    'petchay_dataset' as table_name,
    column_name, 
    data_type, 
    is_nullable
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'petchay_dataset'
AND column_name IN ('label', 'label_confidence', 'image_region', 'bounding_box', 'annotation_notes', 'quality_score', 'is_verified', 'verified_by', 'verified_at', 'updated_at')
ORDER BY column_name;

-- Add updated_at column to petchay_dataset if it doesn't exist (for trigger)
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

-- Summary message
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Migration completed successfully!';
    RAISE NOTICE 'Tables created/updated: yolo_files, petchay_dataset';
    RAISE NOTICE 'New label fields added for accurate detection';
    RAISE NOTICE '========================================';
END $$;

