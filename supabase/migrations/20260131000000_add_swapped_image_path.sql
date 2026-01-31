-- Add swapped_image_path column to simulations table
-- This stores the Face Swap result image for parts-based application
ALTER TABLE simulations ADD COLUMN IF NOT EXISTS swapped_image_path TEXT;

COMMENT ON COLUMN simulations.swapped_image_path IS 'Base64 encoded swapped face image for parts mode';
