-- Add parts_result_image_path column to simulations table
-- This stores the parts blend result image
ALTER TABLE simulations ADD COLUMN IF NOT EXISTS parts_result_image_path TEXT;

COMMENT ON COLUMN simulations.parts_result_image_path IS 'Base64 encoded parts blend result image';
