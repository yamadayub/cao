-- Drop unused generation_jobs table
-- This table was designed for async job queue but was never used.
-- Face swap is handled synchronously via Replicate API.

DROP TABLE IF EXISTS generation_jobs;
DROP FUNCTION IF EXISTS update_generation_jobs_updated_at();
