-- Add encrypted PII columns to profiles table
-- PII fields are encrypted with Fernet (AES-128-CBC) at the application layer

ALTER TABLE profiles
  ADD COLUMN IF NOT EXISTS email_encrypted TEXT,
  ADD COLUMN IF NOT EXISTS phone_encrypted TEXT,
  ADD COLUMN IF NOT EXISTS line_user_id_encrypted TEXT,
  ADD COLUMN IF NOT EXISTS line_display_name TEXT,
  ADD COLUMN IF NOT EXISTS first_name TEXT,
  ADD COLUMN IF NOT EXISTS last_name TEXT,
  ADD COLUMN IF NOT EXISTS clerk_external_accounts JSONB DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS last_sign_in_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS clerk_synced_at TIMESTAMPTZ;

COMMENT ON COLUMN profiles.email_encrypted IS 'Fernet暗号化されたメールアドレス';
COMMENT ON COLUMN profiles.phone_encrypted IS 'Fernet暗号化された電話番号';
COMMENT ON COLUMN profiles.line_user_id_encrypted IS 'Fernet暗号化されたLINE User ID';
COMMENT ON COLUMN profiles.line_display_name IS 'LINE表示名（非PII）';
COMMENT ON COLUMN profiles.clerk_external_accounts IS 'Clerk外部アカウント情報（プロバイダ種類等）';
