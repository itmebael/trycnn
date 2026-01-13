-- Quick fix: Add updated_at column to petchay_dataset if it doesn't exist
-- Run this if you get the error: column "updated_at" does not exist

-- Add updated_at column if it doesn't exist
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
        
        -- Update existing rows to have current timestamp
        UPDATE public.petchay_dataset 
        SET updated_at = created_at 
        WHERE updated_at IS NULL;
        
        RAISE NOTICE 'Added updated_at column to petchay_dataset table';
    ELSE
        RAISE NOTICE 'Column updated_at already exists in petchay_dataset table';
    END IF;
END $$;

-- Create or replace the trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger if it doesn't exist
DROP TRIGGER IF EXISTS update_petchay_dataset_updated_at ON public.petchay_dataset;
CREATE TRIGGER update_petchay_dataset_updated_at
    BEFORE UPDATE ON public.petchay_dataset
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Verify the column was added
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'petchay_dataset'
AND column_name = 'updated_at';

