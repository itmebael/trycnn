-- ============================================================
-- FIX: Drop and recreate match_petchay_embeddings function
-- Fixes "cannot change return type of existing function" error
-- ============================================================

-- Drop function with all possible signatures
DROP FUNCTION IF EXISTS match_petchay_embeddings(VECTOR, FLOAT, INT) CASCADE;
DROP FUNCTION IF EXISTS match_petchay_embeddings(VECTOR, DOUBLE PRECISION, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS match_petchay_embeddings(vector, double precision, integer) CASCADE;
DROP FUNCTION IF EXISTS match_petchay_embeddings(vector(512), float, int) CASCADE;
DROP FUNCTION IF EXISTS match_petchay_embeddings(vector(512), double precision, integer) CASCADE;

-- Drop function by name (catches any remaining variations)
DO $$ 
BEGIN
    -- Try to drop function if it exists
    EXECUTE 'DROP FUNCTION IF EXISTS match_petchay_embeddings CASCADE';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Function may not exist or already dropped: %', SQLERRM;
END $$;

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

-- Add comment
COMMENT ON FUNCTION match_petchay_embeddings IS 
    'Finds similar pechay images using vector similarity search on embeddings. Returns top N matches above threshold.';

-- Verify function was created
SELECT 
    routine_name,
    routine_type,
    data_type as return_type
FROM information_schema.routines
WHERE routine_schema = 'public'
AND routine_name = 'match_petchay_embeddings';

-- Success message
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Function match_petchay_embeddings fixed!';
    RAISE NOTICE '  - Old function dropped';
    RAISE NOTICE '  - New function created with correct return type';
    RAISE NOTICE '========================================';
END $$;

