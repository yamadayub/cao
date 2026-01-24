# Cao - Development/Staging 環境セットアップガイド

## 概要

このガイドでは、Caoの開発・ステージング環境に必要な外部サービスのセットアップ手順を説明します。

### 使用するサービス

| サービス | 用途 | 料金プラン |
|---------|------|-----------|
| GitHub | ソースコード管理、CI/CD | Free |
| Clerk | 認証（Google/LINE OAuth） | Free (10,000 MAU) |
| Supabase | データベース、ストレージ | Free (500MB DB, 1GB Storage) |
| Vercel | フロントエンドホスティング | Free (Hobby) |
| Heroku | バックエンドホスティング | Eco ($5/月) または Railway/Render |

---

## 1. GitHub リポジトリ設定

### 1.1 リポジトリ作成

```bash
# GitHubでリポジトリを作成後
cd /Users/yosuke/dev/cao
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cao.git
git push -u origin main
```

### 1.2 Branch Protection Rules（推奨）

GitHub → Settings → Branches → Add rule:
- Branch name pattern: `main`
- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging

### 1.3 Secrets 設定（後で設定）

GitHub → Settings → Secrets and variables → Actions で以下を設定：
- `CLERK_SECRET_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `VERCEL_TOKEN`
- `HEROKU_API_KEY`

---

## 2. Clerk 認証設定

### 2.1 アカウント作成

1. https://clerk.com にアクセス
2. 「Start building for free」をクリック
3. GitHubまたはGoogleでサインアップ

### 2.2 アプリケーション作成

1. Dashboard → 「Create application」
2. Application name: `Cao Development`
3. Sign-in options:
   - ✅ Email
   - ✅ Google
   - ✅ LINE（後で設定）

### 2.3 Google OAuth 設定

1. [Google Cloud Console](https://console.cloud.google.com) にアクセス
2. 新しいプロジェクトを作成: `cao-dev`
3. APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client IDs
4. Application type: Web application
5. Authorized JavaScript origins:
   - `http://localhost:3000`
   - `https://YOUR_APP.vercel.app`
6. Authorized redirect URIs:
   - `https://hardy-dogfish-46.accounts.dev/v1/oauth_callback` (Clerkから取得)
7. Client ID と Client Secret をコピー
8. Clerk Dashboard → User & Authentication → Social Connections → Google
9. Client ID と Client Secret を貼り付け

### 2.4 LINE Login 設定

1. [LINE Developers Console](https://developers.line.biz/console/) にアクセス
2. Create a new provider: `Cao`
3. Create a new channel → LINE Login
4. Channel name: `Cao Development`
5. App types: ✅ Web app
6. Callback URL:
   - `https://hardy-dogfish-46.accounts.dev/v1/oauth_callback` (Clerkから取得)
7. Email address permission: ✅ Apply
8. Channel ID と Channel Secret をコピー
9. Clerk Dashboard → User & Authentication → Social Connections → LINE
10. Channel ID と Channel Secret を貼り付け

### 2.5 API Keys 取得

Clerk Dashboard → API Keys から以下をコピー：
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` (pk_test_...)
- `CLERK_SECRET_KEY` (sk_test_...)

### 2.6 Frontend 環境変数設定

```bash
# /frontend/.env.local
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_xxxxxxxxxx
CLERK_SECRET_KEY=sk_test_xxxxxxxxxx
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/
```

---

## 3. Supabase 設定

### 3.1 プロジェクト作成

1. https://supabase.com にアクセス
2. 「Start your project」→ GitHubでサインイン
3. 「New project」
   - Organization: 新規作成または既存を選択
   - Project name: `cao-dev`
   - Database Password: 強力なパスワードを生成（保存しておく）
   - Region: Northeast Asia (Tokyo)
4. 「Create new project」をクリック

### 3.2 データベーススキーマ作成

SQL Editor で以下を実行：

```sql
-- Profiles テーブル（Clerkユーザーと連携）
CREATE TABLE profiles (
  id TEXT PRIMARY KEY,  -- Clerk user_id
  display_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Simulations テーブル
CREATE TABLE simulations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL REFERENCES profiles(id),
  current_image_path TEXT NOT NULL,
  ideal_image_path TEXT NOT NULL,
  result_images JSONB DEFAULT '[]',
  settings JSONB DEFAULT '{}',
  share_token TEXT UNIQUE,
  is_public BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- インデックス
CREATE INDEX idx_simulations_user_id ON simulations(user_id);
CREATE INDEX idx_simulations_share_token ON simulations(share_token);

-- RLS (Row Level Security) ポリシー
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;

-- Profiles: 自分のプロフィールのみ操作可能
CREATE POLICY "Users can view own profile" ON profiles
  FOR SELECT USING (id = current_setting('request.jwt.claims', true)::json->>'sub');

CREATE POLICY "Users can update own profile" ON profiles
  FOR UPDATE USING (id = current_setting('request.jwt.claims', true)::json->>'sub');

CREATE POLICY "Users can insert own profile" ON profiles
  FOR INSERT WITH CHECK (id = current_setting('request.jwt.claims', true)::json->>'sub');

-- Simulations: 自分のシミュレーションのみ操作可能、公開シミュレーションは誰でも閲覧可能
CREATE POLICY "Users can view own simulations" ON simulations
  FOR SELECT USING (
    user_id = current_setting('request.jwt.claims', true)::json->>'sub'
    OR is_public = TRUE
    OR share_token IS NOT NULL
  );

CREATE POLICY "Users can insert own simulations" ON simulations
  FOR INSERT WITH CHECK (user_id = current_setting('request.jwt.claims', true)::json->>'sub');

CREATE POLICY "Users can update own simulations" ON simulations
  FOR UPDATE USING (user_id = current_setting('request.jwt.claims', true)::json->>'sub');

CREATE POLICY "Users can delete own simulations" ON simulations
  FOR DELETE USING (user_id = current_setting('request.jwt.claims', true)::json->>'sub');
```

### 3.3 Storage バケット作成

Storage → New bucket:

**1. uploads バケット（プライベート）**
- Name: `uploads`
- Public bucket: ❌ OFF
- File size limit: 10MB
- Allowed MIME types: `image/jpeg, image/png`

**2. results バケット（プライベート）**
- Name: `results`
- Public bucket: ❌ OFF
- File size limit: 50MB
- Allowed MIME types: `image/jpeg, image/png, image/gif`

### 3.4 Storage ポリシー設定

SQL Editor で実行：

```sql
-- uploads バケット: 認証ユーザーのみアップロード可能
CREATE POLICY "Authenticated users can upload" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'uploads'
    AND auth.role() = 'authenticated'
  );

CREATE POLICY "Users can view own uploads" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'uploads'
    AND (storage.foldername(name))[1] = (current_setting('request.jwt.claims', true)::json->>'sub')
  );

-- results バケット: 認証ユーザーのみ
CREATE POLICY "Authenticated users can insert results" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'results'
    AND auth.role() = 'authenticated'
  );

CREATE POLICY "Users can view own results" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'results'
    AND (storage.foldername(name))[1] = (current_setting('request.jwt.claims', true)::json->>'sub')
  );
```

### 3.5 API Keys 取得

Settings → API から以下をコピー：
- `Project URL` (https://xxx.supabase.co)
- `anon/public` key
- `service_role` key（バックエンド用、秘密にする）

### 3.6 環境変数設定

```bash
# /frontend/.env.local に追加
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# /backend/.env
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## 4. Vercel (Frontend) デプロイ設定

### 4.1 Vercel アカウント作成

1. https://vercel.com にアクセス
2. 「Start Deploying」→ GitHubでサインイン

### 4.2 プロジェクトインポート

1. 「Add New」→「Project」
2. GitHubリポジトリ `cao` を選択
3. Configure Project:
   - Framework Preset: Next.js
   - Root Directory: `frontend`
   - Build Command: `pnpm build`
   - Output Directory: `.next`
   - Install Command: `pnpm install`

### 4.3 環境変数設定

Environment Variables に以下を追加：

| Key | Value | Environment |
|-----|-------|-------------|
| NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY | pk_test_xxx | All |
| CLERK_SECRET_KEY | sk_test_xxx | All |
| NEXT_PUBLIC_SUPABASE_URL | https://xxx.supabase.co | All |
| NEXT_PUBLIC_SUPABASE_ANON_KEY | eyJxxx... | All |
| NEXT_PUBLIC_API_URL | https://cao-api-dev.herokuapp.com | All |

### 4.4 デプロイ

1. 「Deploy」をクリック
2. デプロイ完了後、URLをメモ（例: `cao-dev.vercel.app`）

### 4.5 Clerk Allowed Origins 更新

Clerk Dashboard → Paths → Allowed origins に追加：
- `https://cao-dev.vercel.app`

---

## 5. Heroku (Backend) デプロイ設定

### 5.1 Heroku アカウント作成

1. https://heroku.com にアクセス
2. 「Sign up for Free」

### 5.2 Heroku CLI インストール

```bash
# macOS
brew tap heroku/brew && brew install heroku

# ログイン
heroku login
```

### 5.3 アプリ作成

```bash
cd /Users/yosuke/dev/cao/backend

# Herokuアプリ作成
heroku create cao-api-dev

# Python buildpack追加
heroku buildpacks:set heroku/python
```

### 5.4 環境変数設定

```bash
heroku config:set APP_ENV=staging
heroku config:set DEBUG=false
heroku config:set API_VERSION=1.0.0
heroku config:set CORS_ORIGINS=https://cao-dev.vercel.app,http://localhost:3000
heroku config:set SUPABASE_URL=https://xxx.supabase.co
heroku config:set SUPABASE_SERVICE_KEY=eyJxxx...
heroku config:set MAX_IMAGE_SIZE_MB=10
heroku config:set RATE_LIMIT_PER_MINUTE=30
```

### 5.5 Procfile 確認

```bash
# /backend/Procfile
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 5.6 デプロイ

```bash
# backend ディレクトリをサブツリーとしてプッシュ
cd /Users/yosuke/dev/cao
git subtree push --prefix backend heroku main

# または heroku.yml を使う場合
heroku stack:set container
```

### 5.7 確認

```bash
# ログ確認
heroku logs --tail

# ヘルスチェック
curl https://cao-api-dev.herokuapp.com/health
```

---

## 6. GitHub Actions CI/CD 設定

### 6.1 Frontend CI

`.github/workflows/frontend-ci.yml` を作成：

```yaml
name: Frontend CI

on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'
  pull_request:
    branches: [main]
    paths:
      - 'frontend/**'

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend

    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v2
        with:
          version: 8

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
          cache-dependency-path: frontend/pnpm-lock.yaml

      - name: Install dependencies
        run: pnpm install

      - name: Lint
        run: pnpm lint

      - name: Type check
        run: pnpm tsc --noEmit

      - name: Unit tests
        run: pnpm test:unit

      - name: Build
        run: pnpm build
        env:
          NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: ${{ secrets.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY }}
          NEXT_PUBLIC_SUPABASE_URL: ${{ secrets.NEXT_PUBLIC_SUPABASE_URL }}
          NEXT_PUBLIC_SUPABASE_ANON_KEY: ${{ secrets.NEXT_PUBLIC_SUPABASE_ANON_KEY }}
          NEXT_PUBLIC_API_URL: https://cao-api-dev.herokuapp.com
```

### 6.2 Backend CI

`.github/workflows/backend-ci.yml` を作成：

```yaml
name: Backend CI

on:
  push:
    branches: [main]
    paths:
      - 'backend/**'
  pull_request:
    branches: [main]
    paths:
      - 'backend/**'

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov httpx

      - name: Run tests
        run: pytest tests/ -v --cov=app

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Deploy to Heroku
        uses: akhileshns/heroku-deploy@v3.12.14
        with:
          heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
          heroku_app_name: cao-api-dev
          heroku_email: ${{ secrets.HEROKU_EMAIL }}
          appdir: backend
```

### 6.3 E2E Tests

`.github/workflows/e2e.yml` を作成：

```yaml
name: E2E Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  e2e:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend

    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v2
        with:
          version: 8

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
          cache-dependency-path: frontend/pnpm-lock.yaml

      - name: Install dependencies
        run: pnpm install

      - name: Install Playwright browsers
        run: pnpm exec playwright install --with-deps chromium

      - name: Run E2E tests
        run: pnpm test:e2e
        env:
          CI: true

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: frontend/playwright-report/
          retention-days: 7
```

---

## 7. セットアップ完了チェックリスト

### GitHub
- [ ] リポジトリ作成完了
- [ ] main ブランチにプッシュ完了
- [ ] Secrets 設定完了

### Clerk
- [ ] アプリケーション作成完了
- [ ] Google OAuth 設定完了
- [ ] LINE Login 設定完了
- [ ] API Keys 取得完了

### Supabase
- [ ] プロジェクト作成完了
- [ ] データベーススキーマ作成完了
- [ ] Storage バケット作成完了
- [ ] API Keys 取得完了

### Vercel
- [ ] プロジェクトインポート完了
- [ ] 環境変数設定完了
- [ ] デプロイ成功
- [ ] Clerk Allowed Origins 更新完了

### Heroku
- [ ] アプリ作成完了
- [ ] 環境変数設定完了
- [ ] デプロイ成功
- [ ] ヘルスチェック確認完了

### CI/CD
- [ ] frontend-ci.yml 作成完了
- [ ] backend-ci.yml 作成完了
- [ ] e2e.yml 作成完了
- [ ] 最初のCI実行成功

---

## 8. 環境変数まとめ

### Frontend (.env.local)

```bash
# Clerk
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_xxxxx
CLERK_SECRET_KEY=sk_test_xxxxx
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/

# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx

# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend (.env)

```bash
# App
APP_ENV=development
DEBUG=true
API_VERSION=1.0.0

# CORS
CORS_ORIGINS=http://localhost:3000

# Supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx

# Rate Limiting
RATE_LIMIT_PER_MINUTE=30

# Image Processing
MAX_IMAGE_SIZE_MB=10
MAX_IMAGE_DIMENSION=2048
```

---

## 9. トラブルシューティング

### Clerk

**問題**: LINE Login が動作しない
- LINE Developers Console で Callback URL が正しいか確認
- Clerk Dashboard で Channel ID/Secret が正しいか確認

**問題**: 「ClerkProvider not found」エラー
- `.env.local` に `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` が設定されているか確認
- 開発サーバーを再起動

### Supabase

**問題**: RLS でアクセスが拒否される
- JWT の `sub` クレームが正しく設定されているか確認
- ポリシーが正しいか SQL Editor で確認

### Heroku

**問題**: デプロイが失敗する
- `requirements.txt` が正しいか確認
- Python バージョンが `runtime.txt` で指定されているか確認

**問題**: アプリが起動しない
- `heroku logs --tail` でエラーを確認
- 環境変数が正しく設定されているか確認

---

## 次のステップ

1. 本番環境のセットアップ
   - 別のClerkアプリケーション（Production）
   - 別のSupabaseプロジェクト（Production）
   - Vercel Production deployment
   - Heroku Production app

2. ドメイン設定
   - カスタムドメインの取得
   - Vercel/Herokuへのドメイン設定
   - SSL証明書の設定

3. 監視設定
   - Sentry (エラートラッキング)
   - Vercel Analytics
   - Heroku Metrics
