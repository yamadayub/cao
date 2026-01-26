-- ============================================
-- Generation Jobs Table
-- Async job queue for face simulation generation
-- ============================================

-- Create generation_jobs table
CREATE TABLE IF NOT EXISTS generation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Job configuration
    mode TEXT NOT NULL CHECK (mode IN ('morph', 'parts')),
    parts JSONB DEFAULT '[]'::jsonb,           -- For mode='parts': ["eyes", "nose", "lips"]
    strength FLOAT DEFAULT 0.5 CHECK (strength >= 0 AND strength <= 1),
    seed INTEGER,

    -- Input images (Base64 or storage paths)
    base_image_path TEXT NOT NULL,
    target_image_path TEXT NOT NULL,

    -- Output
    result_image_path TEXT,                    -- Result Base64 or storage path

    -- Job status
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    error TEXT,

    -- Metadata
    user_id TEXT,                              -- Optional: link to user
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_user_id ON generation_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_created_at ON generation_jobs(created_at DESC);

-- Index for worker polling: find oldest queued job
CREATE INDEX IF NOT EXISTS idx_generation_jobs_queued ON generation_jobs(created_at ASC)
    WHERE status = 'queued';

-- Updated_at trigger
CREATE OR REPLACE FUNCTION update_generation_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_generation_jobs_updated_at
    BEFORE UPDATE ON generation_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_generation_jobs_updated_at();

-- Comment
COMMENT ON TABLE generation_jobs IS 'Async generation job queue for face morphing/blending';
COMMENT ON COLUMN generation_jobs.mode IS 'Generation mode: morph (full face) or parts (selective)';
COMMENT ON COLUMN generation_jobs.parts IS 'For parts mode: array of part names to blend';
COMMENT ON COLUMN generation_jobs.strength IS 'Blend strength 0-1, higher = more target features';
COMMENT ON COLUMN generation_jobs.status IS 'Job lifecycle: queued -> running -> succeeded/failed';
