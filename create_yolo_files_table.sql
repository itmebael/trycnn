-- Create yolo_files table for tracking YOLO training dataset files
-- This table is used to store metadata about images in the training dataset
-- and is checked during scanning to determine if an image is "known"

CREATE TABLE IF NOT EXISTS public.yolo_files (
    id BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    filename TEXT NULL,
    file_type TEXT NULL,
    dataset_type TEXT NULL,
    url TEXT NULL,
    treatment TEXT NULL,
    uploaded_at TIMESTAMP WITHOUT TIME ZONE NULL DEFAULT NOW(),
    CONSTRAINT yolo_files_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

-- Create index on filename for faster lookups during scanning
CREATE INDEX IF NOT EXISTS idx_yolo_files_filename ON public.yolo_files(filename);

-- Create index on dataset_type for filtering by condition (Healthy/Diseased)
CREATE INDEX IF NOT EXISTS idx_yolo_files_dataset_type ON public.yolo_files(dataset_type);

-- If table already exists, add treatment column (for existing databases)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'yolo_files' 
        AND column_name = 'treatment'
    ) THEN
        ALTER TABLE public.yolo_files ADD COLUMN treatment TEXT NULL;
    END IF;
END $$;

-- Add comment to table
COMMENT ON TABLE public.yolo_files IS 'Stores metadata about images in the YOLO training dataset. Used during scanning to check if an image is known.';

-- Add comments to columns
COMMENT ON COLUMN public.yolo_files.id IS 'Primary key, auto-generated';
COMMENT ON COLUMN public.yolo_files.filename IS 'Unique filename of the image file';
COMMENT ON COLUMN public.yolo_files.file_type IS 'Type of file (e.g., "image")';
COMMENT ON COLUMN public.yolo_files.dataset_type IS 'Dataset category: Healthy, Diseased, or specific disease name';
COMMENT ON COLUMN public.yolo_files.url IS 'URL or path to the image file';
COMMENT ON COLUMN public.yolo_files.treatment IS 'Treatment recommendations or information for the disease/condition';
COMMENT ON COLUMN public.yolo_files.uploaded_at IS 'Timestamp when the file was uploaded';

