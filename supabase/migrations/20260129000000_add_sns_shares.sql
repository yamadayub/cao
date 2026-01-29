-- SNS Shares テーブル
-- シェア画像とメタデータを保存

-- ============================================
-- sns_shares テーブル
-- ============================================
CREATE TABLE IF NOT EXISTS sns_shares (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  simulation_id UUID REFERENCES simulations(id) ON DELETE SET NULL,
  template TEXT NOT NULL CHECK (template IN ('before_after', 'single', 'parts_highlight')),
  caption TEXT CHECK (char_length(caption) <= 140),
  applied_parts JSONB DEFAULT NULL,
  share_image_url TEXT NOT NULL,
  og_image_url TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL
);

COMMENT ON TABLE sns_shares IS 'SNSシェア用の画像とメタデータ';
COMMENT ON COLUMN sns_shares.template IS 'シェア画像テンプレート: before_after, single, parts_highlight';
COMMENT ON COLUMN sns_shares.caption IS 'ユーザーが入力したキャプション（最大140文字）';
COMMENT ON COLUMN sns_shares.applied_parts IS 'パーツハイライトテンプレート用の適用パーツリスト';
COMMENT ON COLUMN sns_shares.expires_at IS 'シェアの有効期限（30日後）';

-- インデックス
CREATE INDEX IF NOT EXISTS idx_sns_shares_user_id ON sns_shares(user_id);
CREATE INDEX IF NOT EXISTS idx_sns_shares_expires_at ON sns_shares(expires_at);
CREATE INDEX IF NOT EXISTS idx_sns_shares_created_at ON sns_shares(created_at DESC);

-- RLS
ALTER TABLE sns_shares ENABLE ROW LEVEL SECURITY;

-- シェアは誰でも閲覧可能（期限切れチェックはアプリ層で行う）
CREATE POLICY "Shares are viewable by everyone"
  ON sns_shares FOR SELECT
  USING (true);

-- ============================================
-- Storage バケット: shares (public)
-- ============================================
-- 注意: Storage バケットはSQLから直接作成できません。
-- Supabase Dashboardから手動で作成してください:
--
-- 1. Supabase Dashboard → Storage → New bucket
-- 2. Bucket name: shares
-- 3. Public bucket: Yes (チェックを入れる)
-- 4. File size limit: 5MB
-- 5. Allowed MIME types: image/png, image/jpeg
--
-- バケットのRLSポリシー（Dashboard → Storage → Policies）:
-- - SELECT: true (誰でも閲覧可能)
-- - INSERT: auth.role() = 'service_role' (バックエンドのみ)
-- - DELETE: auth.role() = 'service_role' (バックエンドのみ)
