# 環境構成ガイド

このドキュメントでは、CAOアプリケーションのStaging環境とProduction環境の構成について説明します。

## 概要

CAOアプリケーションは以下のサービスを利用しています：

- **Vercel**: フロントエンドホスティング
- **Heroku**: バックエンドAPI
- **Supabase**: データベース・ストレージ
- **Clerk**: 認証基盤
- **Google OAuth**: Googleログイン
- **LINE OAuth**: LINEログイン

## 環境構成サマリー

| サービス | Staging | Production | 構成方針 |
|----------|---------|------------|----------|
| Vercel | Preview環境 | Production環境 | 1プロジェクト内で分離 |
| Heroku | `cao-api-staging` | `cao-api-production` | 別アプリとして作成 |
| Supabase | `cao-staging` | `cao-production` | 別プロジェクトとして作成 |
| Clerk | Development インスタンス | Production インスタンス | 1アプリ内で分離 |
| Google OAuth | テスト用クライアント | 本番用クライアント | 1プロジェクト内で分離 |
| LINE OAuth | `cao-staging` チャネル | `cao-production` チャネル | 別チャネルとして作成 |

---

## 1. Vercel（フロントエンド）

### 構成

```
cao-frontend (1プロジェクト)
├── Production: main ブランチ → cao.vercel.app
├── Preview: その他ブランチ → cao-xxx-staging.vercel.app
└── Development: ローカル → localhost:3000
```

### 環境変数設定

Vercelダッシュボード → Settings → Environment Variables で設定：

| 変数名 | Production | Preview | Development |
|--------|------------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `https://cao-api-production.herokuapp.com` | `https://cao-api-staging.herokuapp.com` | `http://localhost:8000` |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | `pk_live_...` | `pk_test_...` | `pk_test_...` |
| `CLERK_SECRET_KEY` | `sk_live_...` | `sk_test_...` | `sk_test_...` |

### デプロイトリガー

- `main` ブランチへのpush → Production環境に自動デプロイ
- その他ブランチへのpush → Preview環境に自動デプロイ

---

## 2. Heroku（バックエンドAPI）

### 構成

```
cao-api-staging     ← 開発・テスト用
cao-api-production  ← 本番用
```

### Heroku Pipeline

```
[cao-api-staging] → Promote → [cao-api-production]
     (staging)                    (production)
```

### セットアップコマンド

```bash
# Production アプリ作成
heroku create cao-api-production

# Pipeline 作成・設定
heroku pipelines:create cao-api
heroku pipelines:add cao-api --app cao-api-staging --stage staging
heroku pipelines:add cao-api --app cao-api-production --stage production
```

### 環境変数設定

各アプリで以下を設定（Heroku Dashboard → Settings → Config Vars）：

| 変数名 | Staging | Production |
|--------|---------|------------|
| `SUPABASE_URL` | Staging用URL | Production用URL |
| `SUPABASE_ANON_KEY` | Staging用キー | Production用キー |
| `SUPABASE_SERVICE_ROLE_KEY` | Staging用キー | Production用キー |
| `CLERK_SECRET_KEY` | `sk_test_...` | `sk_live_...` |
| `CORS_ORIGINS` | Staging用Vercel URL | Production用Vercel URL |

### デプロイ方法

```bash
# Stagingへデプロイ
git push heroku-staging main

# ProductionへPromote（Heroku Dashboard または CLI）
heroku pipelines:promote --app cao-api-staging
```

---

## 3. Supabase（データベース・ストレージ）

### 構成

```
cao-staging     ← 開発・テスト用DB
cao-production  ← 本番用DB
```

### 分離の理由

- DBスキーマの変更をStagingで先にテスト可能
- 本番データとテストデータの完全分離
- 誤操作による本番データ破損を防止
- Storage bucketも環境ごとに分離

### マイグレーション手順

1. Stagingでマイグレーション実行・動作確認
2. 問題なければProductionに同じマイグレーションを適用

```bash
# Staging
supabase db push --project-ref <staging-ref>

# Production（Staging確認後）
supabase db push --project-ref <production-ref>
```

### 環境変数

| 変数名 | 取得場所 |
|--------|----------|
| `SUPABASE_URL` | Project Settings → API → Project URL |
| `SUPABASE_ANON_KEY` | Project Settings → API → anon public |
| `SUPABASE_SERVICE_ROLE_KEY` | Project Settings → API → service_role |

---

## 4. Clerk（認証基盤）

### 構成

```
cao (1アプリケーション)
├── Development インスタンス ← Staging用
└── Production インスタンス  ← 本番用
```

### 特徴

- 1つのClerkアプリ内にDevelopment/Productionが自動作成される
- APIキーは各インスタンスで異なる
- Social connections（Google, LINE）は各インスタンスで個別設定が必要

### 設定場所

Clerk Dashboard → 該当アプリ → 左上のインスタンス切り替えで Dev/Prod を選択

### APIキー取得

Dashboard → Developers → API Keys で確認：

- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`: フロントエンド用（公開可）
- `CLERK_SECRET_KEY`: バックエンド用（秘密）

### Social Connections設定

各インスタンスで個別に設定が必要：

1. Dashboard → User & Authentication → Social connections
2. Google, LINE それぞれを有効化
3. 各OAuthプロバイダーの認証情報を設定

---

## 5. Google OAuth

### 構成

```
Google Cloud Project: cao-app
├── OAuth Client: cao-staging
│   └── Redirect URI: https://clerk.<dev-domain>.accounts.dev/v1/oauth_callback
└── OAuth Client: cao-production
    └── Redirect URI: https://clerk.<prod-domain>.accounts.com/v1/oauth_callback
```

### 設定手順

1. [Google Cloud Console](https://console.cloud.google.com/) にアクセス
2. APIs & Services → Credentials
3. Create Credentials → OAuth client ID
4. Application type: Web application
5. Authorized redirect URIs: Clerkダッシュボードで確認したURLを設定

### Clerkへの設定

1. Clerk Dashboard → Social connections → Google
2. Client ID と Client Secret を入力
3. Development/Production 両方で設定

---

## 6. LINE OAuth

### 構成

```
LINE Developers Provider: CAO
├── cao-staging (Developing状態)
│   └── Callback URL: Clerk Development用
└── cao-production (本番公開時にPublished)
    └── Callback URL: Clerk Production用
```

### 運用方針

| 環境 | LINEチャネル状態 | アクセス可能ユーザー |
|------|------------------|---------------------|
| Staging | Developing | 開発者・テスターのみ |
| Production（テスト中） | Developing | 登録テスターのみ |
| Production（本番） | Published | 全ユーザー |

### 設定手順

1. [LINE Developers Console](https://developers.line.biz/console/) にアクセス
2. Provider選択 → Create a new channel → LINE Login
3. Channel名: `cao-staging` または `cao-production`
4. Callback URL: Clerkダッシュボードで確認したURLを設定

### Clerkへの設定

1. Clerk Dashboard → Social connections → LINE
2. Channel ID と Channel Secret を入力
3. Development/Production 両方で設定

---

## 環境変数一覧

### フロントエンド（Vercel）

| 変数名 | Staging (Preview) | Production |
|--------|-------------------|------------|
| `NEXT_PUBLIC_API_URL` | `https://cao-api-staging.herokuapp.com` | `https://cao-api-production.herokuapp.com` |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | `pk_test_...` | `pk_live_...` |
| `CLERK_SECRET_KEY` | `sk_test_...` | `sk_live_...` |
| `NEXT_PUBLIC_CLERK_SIGN_IN_URL` | `/sign-in` | `/sign-in` |
| `NEXT_PUBLIC_CLERK_SIGN_UP_URL` | `/sign-up` | `/sign-up` |

### バックエンド（Heroku）

| 変数名 | Staging | Production |
|--------|---------|------------|
| `SUPABASE_URL` | `https://xxx-staging.supabase.co` | `https://xxx-production.supabase.co` |
| `SUPABASE_ANON_KEY` | Staging用 | Production用 |
| `SUPABASE_SERVICE_ROLE_KEY` | Staging用 | Production用 |
| `CLERK_SECRET_KEY` | `sk_test_...` | `sk_live_...` |
| `CORS_ORIGINS` | `https://cao-staging.vercel.app` | `https://cao.vercel.app` |
| `REPLICATE_API_TOKEN` | 共通で使用可 | 共通で使用可 |

---

## デプロイフロー

```
[ローカル開発]
     │
     ▼
[feature ブランチ push]
     │
     ├──→ Vercel: Preview自動デプロイ（確認用）
     │
     ▼
[develop ブランチ merge]
     │
     ├──→ Vercel: Preview自動デプロイ（Staging）
     ├──→ Heroku: cao-api-staging 自動デプロイ
     │
     ▼
[Staging環境でテスト]
     │
     ▼
[main ブランチ merge]
     │
     ├──→ Vercel: Production自動デプロイ
     ├──→ Heroku: Pipeline promote（手動）
     │
     ▼
[Production環境]
```

---

## トラブルシューティング

### Clerk認証エラー

1. 環境変数が正しいインスタンス（Dev/Prod）のものか確認
2. Social connectionsが有効になっているか確認
3. OAuthプロバイダーのCallback URLが正しいか確認

### CORS エラー

1. Herokuの `CORS_ORIGINS` にフロントエンドURLが含まれているか確認
2. Vercelのデプロイ先URLが変わっていないか確認

### データベース接続エラー

1. Supabaseの環境変数が正しいプロジェクトのものか確認
2. Supabaseプロジェクトがpause状態になっていないか確認

### LINE ログインエラー

- "this channel is now developing status" → テスター登録またはチャネル公開が必要
- "invalid_request" → Callback URLの不一致

---

## 関連ドキュメント

- [Vercel Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)
- [Heroku Pipelines](https://devcenter.heroku.com/articles/pipelines)
- [Supabase CLI](https://supabase.com/docs/guides/cli)
- [Clerk Documentation](https://clerk.com/docs)
- [LINE Login Documentation](https://developers.line.biz/ja/docs/line-login/)
