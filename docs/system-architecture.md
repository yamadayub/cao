# Cao - System Architecture

## Overview

CaoはAI顔分析とモーフィングシミュレーションを提供するWebアプリケーション。フロントエンドとバックエンドを分離し、画像処理はPython専用サーバーで実行する構成。

---

## System Diagram
```
┌─────────────────────────────────────────────────────────┐
│                     Cloudflare                          │
│                   (DNS + CDN)                           │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌──────────────────┐              ┌──────────────────────┐
│   Vercel         │              │   Heroku             │
│   (Frontend)     │    REST API  │   (Backend)          │
│                  │◄────────────►│                      │
│   cao.app        │              │   api.cao.app        │
│                  │              │                      │
│   - Next.js 14   │              │   - FastAPI          │
│   - App Router   │              │   - Python 3.11+     │
│   - React 18     │              │   - OpenCV           │
│   - TailwindCSS  │              │   - MediaPipe        │
│   - shadcn/ui    │              │   - NumPy/Pillow     │
└──────────────────┘              └──────────────────────┘
          │                                 │
          │                                 │
          └────────────┬────────────────────┘
                       ▼
             ┌──────────────────────┐
             │   Supabase           │
             │                      │
             │   - Auth             │
             │   - PostgreSQL       │
             │   - Storage          │
             │   - Edge Functions   │
             └──────────────────────┘
```

---

## Technology Stack

### Frontend (Vercel)

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| Framework | Next.js | 14.x | App Router, SSR/SSG |
| UI Library | React | 18.x | Component-based UI |
| Styling | TailwindCSS | 3.x | Utility-first CSS |
| Components | shadcn/ui | latest | Accessible UI components |
| State | Zustand | 4.x | Client state management |
| Forms | React Hook Form + Zod | - | Form validation |
| HTTP Client | ky | 1.x | Lightweight fetch wrapper |
| Testing | Vitest + Playwright | - | Unit/E2E testing |

### Backend (Heroku)

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| Framework | FastAPI | 0.110+ | Async REST API |
| Runtime | Python | 3.11+ | Image processing support |
| Image Processing | OpenCV | 4.9+ | Image manipulation |
| Face Detection | MediaPipe | 0.10+ | Face landmark detection |
| Image Library | Pillow | 10.x | Image I/O, GIF generation |
| Numerical | NumPy | 1.26+ | Array operations |
| Validation | Pydantic | 2.x | Request/Response validation |
| Testing | pytest + httpx | - | API testing |

### Database & Storage (Supabase)

| Service | Purpose |
|---------|---------|
| Auth | User authentication (Email, Google, LINE) |
| PostgreSQL | User data, simulation history, settings |
| Storage | Image storage (uploads, results) |
| Edge Functions | Webhooks, background jobs (optional) |

---

## Repository Structure

### Monorepo構成
```
cao/
├── CLAUDE.md                    # Claude Code設定
├── docs/
│   ├── product-concept.md       # プロダクトコンセプト
│   ├── system-architecture.md   # このドキュメント
│   ├── business-spec.md         # 業務仕様書
│   └── functional-spec.md       # 機能要件書
│
├── frontend/                    # Next.js アプリケーション
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── src/
│   │   ├── app/                 # App Router pages
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   ├── (auth)/          # Auth routes
│   │   │   ├── (app)/           # Protected routes
│   │   │   └── api/             # API routes (if needed)
│   │   ├── components/          # React components
│   │   │   ├── ui/              # shadcn/ui components
│   │   │   └── features/        # Feature components
│   │   ├── lib/                 # Utilities
│   │   │   ├── supabase/        # Supabase client
│   │   │   └── api/             # Backend API client
│   │   ├── hooks/               # Custom hooks
│   │   └── types/               # TypeScript types
│   └── tests/
│       ├── unit/                # Vitest unit tests
│       └── e2e/                 # Playwright E2E tests
│
├── backend/                     # FastAPI アプリケーション
│   ├── pyproject.toml           # Poetry/uv dependency
│   ├── Procfile                 # Heroku config
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Settings
│   │   ├── routers/             # API endpoints
│   │   │   ├── analyze.py
│   │   │   ├── morph.py
│   │   │   └── health.py
│   │   ├── services/            # Business logic
│   │   │   ├── face_detection.py
│   │   │   ├── morphing.py
│   │   │   └── image_processing.py
│   │   ├── models/              # Pydantic models
│   │   └── utils/               # Utilities
│   └── tests/
│       ├── unit/
│       └── integration/
│
├── supabase/                    # Supabase configuration
│   ├── migrations/              # Database migrations
│   ├── seed.sql                 # Seed data
│   └── config.toml              # Local dev config
│
└── .github/
    └── workflows/
        ├── frontend-ci.yml      # Frontend CI/CD
        ├── backend-ci.yml       # Backend CI/CD
        └── e2e.yml              # E2E tests
```

---

## API Design

### Base URL
- Production: `https://api.cao.app`
- Staging: `https://api-staging.cao.app`
- Local: `http://localhost:8000`

### Endpoints

#### Health Check
```
GET /health
Response: { "status": "ok", "version": "1.0.0" }
```

#### Face Analysis
```
POST /api/v1/analyze
Content-Type: multipart/form-data

Request:
  - image: File (JPEG/PNG, max 10MB)

Response:
{
  "success": true,
  "data": {
    "landmarks": [...],           # 478 facial landmarks
    "face_region": {
      "x": 100, "y": 50,
      "width": 400, "height": 500
    },
    "parts": {
      "left_eye": {...},
      "right_eye": {...},
      "nose": {...},
      "lips": {...},
      "face_oval": {...}
    }
  }
}
```

#### Basic Morphing
```
POST /api/v1/morph
Content-Type: multipart/form-data

Request:
  - current_image: File
  - ideal_image: File
  - progress: float (0.0 - 1.0, default: 0.5)

Response:
{
  "success": true,
  "data": {
    "image": "base64_encoded_image",
    "format": "png"
  }
}
```

#### Staged Morphing (複数段階)
```
POST /api/v1/morph/stages
Content-Type: multipart/form-data

Request:
  - current_image: File
  - ideal_image: File
  - stages: list[float] (default: [0, 0.25, 0.5, 0.75, 1.0])

Response:
{
  "success": true,
  "data": {
    "images": [
      { "progress": 0.0, "image": "base64..." },
      { "progress": 0.25, "image": "base64..." },
      ...
    ]
  }
}
```

#### Part-by-Part Morphing
```
POST /api/v1/morph/parts
Content-Type: multipart/form-data

Request:
  - current_image: File
  - ideal_image: File
  - parts: JSON string
    {
      "eyes": 0.5,
      "eyebrows": 0.25,
      "nose": 0.75,
      "lips": 0.5,
      "face_oval": 0.0
    }

Response:
{
  "success": true,
  "data": {
    "image": "base64_encoded_image",
    "format": "png"
  }
}
```

#### GIF Animation
```
POST /api/v1/morph/animation
Content-Type: multipart/form-data

Request:
  - current_image: File
  - ideal_image: File
  - frames: int (default: 30)
  - duration: int (ms per frame, default: 80)
  - loop: bool (default: true)

Response:
{
  "success": true,
  "data": {
    "image": "base64_encoded_gif",
    "format": "gif",
    "frames": 30,
    "duration_ms": 2400
  }
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "FACE_NOT_DETECTED",
    "message": "No face detected in the uploaded image",
    "details": {}
  }
}
```

### Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid request parameters |
| FACE_NOT_DETECTED | 400 | No face found in image |
| MULTIPLE_FACES | 400 | Multiple faces detected |
| IMAGE_TOO_LARGE | 413 | Image exceeds size limit |
| PROCESSING_ERROR | 500 | Internal processing error |
| RATE_LIMITED | 429 | Too many requests |

---

## Database Schema

### Users (Supabase Auth)
Supabase Authのデフォルトテーブルを使用

### Profiles
```sql
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id),
  display_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Simulations
```sql
CREATE TABLE simulations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  current_image_path TEXT NOT NULL,
  ideal_image_path TEXT NOT NULL,
  result_image_path TEXT,
  settings JSONB DEFAULT '{}',
  share_token TEXT UNIQUE,
  is_public BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_simulations_user_id ON simulations(user_id);
CREATE INDEX idx_simulations_share_token ON simulations(share_token);
```

### Storage Buckets
```
- avatars/           # User avatars (public)
- uploads/           # Uploaded images (private)
  - {user_id}/current/
  - {user_id}/ideal/
- results/           # Generated results (private)
  - {user_id}/{simulation_id}/
```

---

## Environment Variables

### Frontend (.env.local)
```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...

# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000

# Analytics (optional)
NEXT_PUBLIC_GA_ID=G-XXXXXXX
```

### Backend (.env)
```bash
# App
APP_ENV=development
DEBUG=true
API_VERSION=1.0.0

# CORS
CORS_ORIGINS=http://localhost:3000,https://cao.app

# Supabase (for storage access)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# Rate Limiting
RATE_LIMIT_PER_MINUTE=30

# Image Processing
MAX_IMAGE_SIZE_MB=10
MAX_IMAGE_DIMENSION=2048
```

---

## Deployment

### Frontend (Vercel)

Vercelは自動デプロイが有効。mainブランチへのpushで自動的にデプロイされる。

```yaml
# vercel.json
{
  "buildCommand": "cd frontend && pnpm build",
  "outputDirectory": "frontend/.next",
  "framework": "nextjs"
}
```

**手動デプロイ:**
```bash
# Vercel CLIを使用
cd frontend
vercel --prod
```

### Backend (Heroku)

このプロジェクトはモノレポ構成のため、`git subtree push`を使用してbackendディレクトリのみをHerokuにデプロイする。

**初回セットアップ:**
```bash
# Heroku CLIでログイン
heroku login

# アプリを作成（既存の場合はスキップ）
heroku create cao-api-dev

# Herokuリモートを追加
heroku git:remote -a cao-api-dev

# ビルドパックを設定
heroku buildpacks:add heroku-community/apt
heroku buildpacks:add heroku/python
```

**デプロイコマンド:**
```bash
# backendディレクトリをHerokuにプッシュ
git subtree push --prefix backend heroku main

# 強制プッシュが必要な場合（履歴の不一致時）
git push heroku `git subtree split --prefix backend main`:main --force
```

**Herokuの設定ファイル（backend/内）:**
```
# backend/Procfile
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT

# backend/runtime.txt
python-3.11.11

# backend/Aptfile (OpenCV依存ライブラリ)
libgl1
libglib2.0-0
libsm6
libxrender1
libxext6
```

**環境変数の設定:**
```bash
heroku config:set APP_ENV=production
heroku config:set CORS_ORIGINS=https://cao.app,https://cao-coral.vercel.app
heroku config:set SUPABASE_URL=https://xxx.supabase.co
heroku config:set SUPABASE_SERVICE_KEY=xxx
```

### CI/CD Pipeline
```
Push to main
    │
    ├─► Frontend CI (GitHub Actions)
    │   ├── pnpm install
    │   ├── pnpm lint
    │   ├── pnpm test:unit
    │   ├── pnpm build
    │   └── Deploy to Vercel (自動)
    │
    └─► Backend CI (GitHub Actions)
        ├── pip install
        ├── ruff check (lint)
        ├── pytest tests/
        └── Deploy to Heroku (手動: git subtree push)
```

**完全なデプロイフロー:**
```bash
# 1. コードをコミット
git add .
git commit -m "Your commit message"

# 2. GitHubにプッシュ（CI実行、Vercel自動デプロイ）
git push origin main

# 3. Herokuにデプロイ（backendのみ）
git subtree push --prefix backend heroku main
```

---

## Performance Considerations

### Image Processing
- 最大画像サイズ: 10MB
- 処理前にリサイズ: 最大2048px
- 結果画像の圧縮: JPEG quality 85%

### Caching
- 顔分析結果: Redis/メモリキャッシュ（5分）
- 生成画像: Supabase Storageに保存

### Rate Limiting
- 未認証: 10 requests/minute
- 認証済み: 30 requests/minute
- Pro: 100 requests/minute

---

## Security

### API Security
- CORS: フロントエンドドメインのみ許可
- Rate Limiting: IP/User単位で制限
- Input Validation: Pydanticで厳格に検証
- File Validation: Magic bytes検証、サイズ制限

### Image Privacy
- アップロード画像: ユーザー別バケットに保存
- 署名付きURL: 一時的なアクセスのみ許可
- 自動削除: 未使用画像は30日後に削除

### Authentication
- Supabase Auth: JWT-based
- API認証: Bearer token（オプション）
- 匿名利用: 機能制限付きで許可

---

## Monitoring

### Frontend
- Vercel Analytics
- Sentry (Error tracking)

### Backend
- Heroku Metrics
- Sentry (Error tracking)
- Custom logging (structured JSON)

### Alerts
- Error rate > 5%
- Response time > 10s (p95)
- Dyno restart