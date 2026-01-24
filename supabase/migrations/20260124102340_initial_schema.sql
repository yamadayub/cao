-- Cao Initial Schema
-- 認証: Clerk (外部), データベース/ストレージ: Supabase

-- ============================================
-- Profiles テーブル（Clerkユーザーと連携）
-- ============================================
CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY,  -- Clerk user_id (user_xxx形式)
  display_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

COMMENT ON TABLE profiles IS 'Clerkユーザーに対応するプロフィール情報';
COMMENT ON COLUMN profiles.id IS 'Clerk user_id';

-- ============================================
-- Simulations テーブル
-- ============================================
CREATE TABLE IF NOT EXISTS simulations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  current_image_path TEXT NOT NULL,
  ideal_image_path TEXT NOT NULL,
  result_images JSONB DEFAULT '[]'::jsonb NOT NULL,
  settings JSONB DEFAULT '{}'::jsonb NOT NULL,
  share_token TEXT UNIQUE,
  is_public BOOLEAN DEFAULT FALSE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

COMMENT ON TABLE simulations IS 'ユーザーのシミュレーション履歴';
COMMENT ON COLUMN simulations.result_images IS '各段階のモーフィング結果画像 [{progress: 0.0, image_path: "..."}]';
COMMENT ON COLUMN simulations.share_token IS '共有用の一意トークン';

-- インデックス
CREATE INDEX IF NOT EXISTS idx_simulations_user_id ON simulations(user_id);
CREATE INDEX IF NOT EXISTS idx_simulations_share_token ON simulations(share_token) WHERE share_token IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_simulations_created_at ON simulations(created_at DESC);

-- ============================================
-- 更新日時の自動更新トリガー
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER simulations_updated_at
  BEFORE UPDATE ON simulations
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

-- ============================================
-- Row Level Security (RLS)
-- ============================================
-- 注意: Clerkを使用しているため、Supabase Authの代わりに
-- サービスロールキーでバックエンドからアクセスします。
-- RLSはバックエンドで適用するか、Supabase Edge Functionsで
-- Clerk JWTを検証する場合に使用します。

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;

-- サービスロール用ポリシー（バックエンドからのアクセス用）
-- サービスロールはRLSをバイパスするため、これらのポリシーは
-- 主にEdge FunctionsやクライアントSDK経由のアクセス用

-- Profiles: 公開読み取り可能（アバター表示用）
CREATE POLICY "Profiles are viewable by everyone"
  ON profiles FOR SELECT
  USING (true);

-- Simulations: 公開または共有トークン付きは誰でも閲覧可能
CREATE POLICY "Public simulations are viewable by everyone"
  ON simulations FOR SELECT
  USING (is_public = true OR share_token IS NOT NULL);

-- ============================================
-- Storage バケット設定
-- ============================================
-- 注意: Storage バケットはCLIから作成できないため、
-- Supabase Dashboardから手動で作成するか、
-- supabase/config.toml で設定してローカル開発用に使用

-- 必要なバケット:
-- 1. uploads (private) - ユーザーがアップロードした画像
-- 2. results (private) - 生成されたシミュレーション結果
