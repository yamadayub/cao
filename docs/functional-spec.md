# Cao - 機能要件書

## 1. 概要

### 1.1 目的
本ドキュメントはCaoシステムの機能要件を定義する。業務仕様書（business-spec.md）で定義されたユースケースを実装可能なレベルまで詳細化し、API仕様、画面仕様、データモデル、非機能要件を網羅する。

### 1.2 対象範囲
- MVP (Phase 1) の全機能
- Phase 2以降の機能は本ドキュメントの対象外

### 1.3 参照ドキュメント
- `/docs/business-spec.md` - 業務仕様書
- `/docs/system-architecture.md` - システムアーキテクチャ

---

## 2. API仕様

### 2.1 共通仕様

#### Base URL
| 環境 | URL |
|------|-----|
| Production | `https://api.cao.app` |
| Staging | `https://api-staging.cao.app` |
| Local | `http://localhost:8000` |

#### 認証
- Bearer Token認証（Supabase JWT）
- 未認証アクセスも許可（機能制限あり）

#### 共通ヘッダー
```
Content-Type: application/json または multipart/form-data
Authorization: Bearer <jwt_token> (認証時のみ)
X-Request-ID: <uuid> (トレーシング用)
```

#### 共通レスポンス形式

**成功時**
```typescript
interface SuccessResponse<T> {
  success: true;
  data: T;
  meta?: {
    request_id: string;
    processing_time_ms: number;
  };
}
```

**エラー時**
```typescript
interface ErrorResponse {
  success: false;
  error: {
    code: ErrorCode;
    message: string;
    details?: Record<string, unknown>;
  };
}

type ErrorCode =
  | 'VALIDATION_ERROR'
  | 'FACE_NOT_DETECTED'
  | 'MULTIPLE_FACES'
  | 'IMAGE_TOO_LARGE'
  | 'INVALID_IMAGE_FORMAT'
  | 'PROCESSING_ERROR'
  | 'RATE_LIMITED'
  | 'UNAUTHORIZED'
  | 'NOT_FOUND'
  | 'INTERNAL_ERROR';
```

#### HTTPステータスコード
| コード | 意味 | 使用ケース |
|--------|------|-----------|
| 200 | 成功 | 正常なレスポンス |
| 400 | Bad Request | バリデーションエラー、顔検出失敗 |
| 401 | Unauthorized | 認証が必要な操作を未認証で実行 |
| 404 | Not Found | リソースが存在しない |
| 413 | Payload Too Large | 画像サイズ超過 |
| 429 | Too Many Requests | レート制限超過 |
| 500 | Internal Server Error | サーバー内部エラー |

---

### 2.2 エンドポイント一覧

| エンドポイント | メソッド | 認証 | 説明 |
|---------------|---------|------|------|
| `/health` | GET | 不要 | ヘルスチェック |
| `/api/v1/analyze` | POST | 不要 | 顔分析 |
| `/api/v1/morph` | POST | 不要 | 単一モーフィング |
| `/api/v1/morph/stages` | POST | 不要 | 段階的モーフィング |
| `/api/v1/simulations` | POST | 必要 | シミュレーション保存 |
| `/api/v1/simulations` | GET | 必要 | シミュレーション一覧取得 |
| `/api/v1/simulations/{id}` | GET | 必要 | シミュレーション詳細取得 |
| `/api/v1/simulations/{id}` | DELETE | 必要 | シミュレーション削除 |
| `/api/v1/simulations/{id}/share` | POST | 必要 | 共有URL生成 |
| `/api/v1/shared/{token}` | GET | 不要 | 共有シミュレーション取得 |
| `/api/v1/blend/parts` | POST | 不要 | パーツ別ブレンド |

---

### 2.3 エンドポイント詳細

#### 2.3.1 ヘルスチェック

```
GET /health
```

**リクエスト**: なし

**レスポンス**
```typescript
interface HealthResponse {
  success: true;
  data: {
    status: 'ok' | 'degraded';
    version: string;
    timestamp: string; // ISO 8601
  };
}
```

**例**
```json
{
  "success": true,
  "data": {
    "status": "ok",
    "version": "1.0.0",
    "timestamp": "2025-01-23T10:00:00Z"
  }
}
```

---

#### 2.3.2 顔分析

```
POST /api/v1/analyze
Content-Type: multipart/form-data
```

**リクエスト**
| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| image | File | Yes | 分析対象画像（JPEG/PNG、最大10MB） |

**レスポンス**
```typescript
interface AnalyzeResponse {
  success: true;
  data: {
    face_detected: boolean;
    face_count: number;
    face_region: {
      x: number;
      y: number;
      width: number;
      height: number;
    } | null;
    landmarks: FaceLandmark[] | null;
    image_info: {
      width: number;
      height: number;
      format: 'jpeg' | 'png';
    };
  };
}

interface FaceLandmark {
  index: number;
  x: number;  // 正規化座標 (0.0 - 1.0)
  y: number;
  z: number;
}
```

**エラーケース**
| エラーコード | 条件 | メッセージ |
|-------------|------|-----------|
| VALIDATION_ERROR | ファイルなし | Image file is required |
| INVALID_IMAGE_FORMAT | 非対応形式 | Only JPEG and PNG formats are supported |
| IMAGE_TOO_LARGE | 10MB超過 | Image size must be under 10MB |
| FACE_NOT_DETECTED | 顔未検出 | No face detected in the uploaded image |
| MULTIPLE_FACES | 複数顔 | Multiple faces detected. Please upload an image with a single face |

---

#### 2.3.3 単一モーフィング

```
POST /api/v1/morph
Content-Type: multipart/form-data
```

**リクエスト**
| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| current_image | File | Yes | - | 現在の顔画像 |
| ideal_image | File | Yes | - | 理想の顔画像 |
| progress | float | No | 0.5 | 変化度合い (0.0 - 1.0) |

**レスポンス**
```typescript
interface MorphResponse {
  success: true;
  data: {
    image: string;  // Base64エンコード
    format: 'png';
    progress: number;
    dimensions: {
      width: number;
      height: number;
    };
  };
}
```

**エラーケース**
| エラーコード | 条件 | メッセージ |
|-------------|------|-----------|
| VALIDATION_ERROR | progressが範囲外 | Progress must be between 0.0 and 1.0 |
| FACE_NOT_DETECTED | いずれかの画像で顔未検出 | No face detected in {current/ideal} image |
| PROCESSING_ERROR | 処理失敗 | Failed to generate morphed image |

---

#### 2.3.4 段階的モーフィング

```
POST /api/v1/morph/stages
Content-Type: multipart/form-data
```

**リクエスト**
| フィールド | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| current_image | File | Yes | - | 現在の顔画像 |
| ideal_image | File | Yes | - | 理想の顔画像 |
| stages | string (JSON array) | No | [0, 0.25, 0.5, 0.75, 1.0] | 生成する段階 |

**レスポンス**
```typescript
interface StagedMorphResponse {
  success: true;
  data: {
    images: Array<{
      progress: number;
      image: string;  // Base64エンコード
    }>;
    format: 'png';
    dimensions: {
      width: number;
      height: number;
    };
  };
}
```

**例**
```json
{
  "success": true,
  "data": {
    "images": [
      { "progress": 0.0, "image": "data:image/png;base64,..." },
      { "progress": 0.25, "image": "data:image/png;base64,..." },
      { "progress": 0.5, "image": "data:image/png;base64,..." },
      { "progress": 0.75, "image": "data:image/png;base64,..." },
      { "progress": 1.0, "image": "data:image/png;base64,..." }
    ],
    "format": "png",
    "dimensions": { "width": 512, "height": 512 }
  }
}
```

---

#### 2.3.5 シミュレーション保存

```
POST /api/v1/simulations
Authorization: Bearer <token>
Content-Type: application/json
```

**リクエスト**
```typescript
interface CreateSimulationRequest {
  current_image: string;  // Base64またはStorage URL
  ideal_image: string;
  result_images: Array<{
    progress: number;
    image: string;
  }>;
  settings?: {
    selected_progress?: number;
    notes?: string;
  };
}
```

**レスポンス**
```typescript
interface SimulationResponse {
  success: true;
  data: {
    id: string;  // UUID
    user_id: string;
    current_image_url: string;
    ideal_image_url: string;
    result_images: Array<{
      progress: number;
      url: string;
    }>;
    settings: Record<string, unknown>;
    share_token: string | null;
    is_public: boolean;
    created_at: string;
    updated_at: string;
  };
}
```

**エラーケース**
| エラーコード | 条件 | メッセージ |
|-------------|------|-----------|
| UNAUTHORIZED | 未認証 | Authentication required |
| VALIDATION_ERROR | 必須フィールド不足 | {field} is required |

---

#### 2.3.6 シミュレーション一覧取得

```
GET /api/v1/simulations
Authorization: Bearer <token>
```

**クエリパラメータ**
| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| limit | int | 20 | 取得件数（最大100） |
| offset | int | 0 | オフセット |
| sort | string | created_at:desc | ソート順 |

**レスポンス**
```typescript
interface SimulationListResponse {
  success: true;
  data: {
    simulations: SimulationSummary[];
    pagination: {
      total: number;
      limit: number;
      offset: number;
      has_more: boolean;
    };
  };
}

interface SimulationSummary {
  id: string;
  thumbnail_url: string;  // result_images[2] (50%)のサムネイル
  created_at: string;
  is_public: boolean;
}
```

---

#### 2.3.7 シミュレーション詳細取得

```
GET /api/v1/simulations/{id}
Authorization: Bearer <token>
```

**レスポンス**: SimulationResponse（2.3.5と同じ）

**エラーケース**
| エラーコード | 条件 | メッセージ |
|-------------|------|-----------|
| NOT_FOUND | 存在しない | Simulation not found |
| UNAUTHORIZED | 他ユーザーのデータ | Access denied |

---

#### 2.3.8 シミュレーション削除

```
DELETE /api/v1/simulations/{id}
Authorization: Bearer <token>
```

**レスポンス**
```typescript
interface DeleteResponse {
  success: true;
  data: {
    deleted: true;
    id: string;
  };
}
```

---

#### 2.3.9 共有URL生成

```
POST /api/v1/simulations/{id}/share
Authorization: Bearer <token>
```

**リクエスト**: なし

**レスポンス**
```typescript
interface ShareResponse {
  success: true;
  data: {
    share_token: string;
    share_url: string;  // https://cao.app/s/{token}
    expires_at: null;   // 無期限
  };
}
```

---

#### 2.3.10 共有シミュレーション取得

```
GET /api/v1/shared/{token}
```

**レスポンス**
```typescript
interface SharedSimulationResponse {
  success: true;
  data: {
    result_images: Array<{
      progress: number;
      url: string;
    }>;
    created_at: string;
  };
}
```

**エラーケース**
| エラーコード | 条件 | メッセージ |
|-------------|------|-----------|
| NOT_FOUND | 無効なトークン | Shared simulation not found |

---

#### 2.3.11 パーツ別ブレンド

```
POST /api/v1/blend/parts
Content-Type: multipart/form-data
```

**リクエスト**
| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| current_image | File | Yes | 現在の顔画像（ベース） |
| ideal_image | File | Yes | 理想の顔画像（パーツ提供元） |
| parts | string (JSON) | Yes | 適用するパーツの指定 |

**partsフィールド形式**
```typescript
interface PartsSelection {
  left_eye: boolean;      // 左目
  right_eye: boolean;     // 右目
  left_eyebrow: boolean;  // 左眉
  right_eyebrow: boolean; // 右眉
  nose: boolean;          // 鼻
  lips: boolean;          // 口
}
```

**レスポンス**
```typescript
interface PartsBlendResponse {
  success: true;
  data: {
    image: string;  // Base64エンコード
    format: 'png';
    applied_parts: string[];  // 実際に適用されたパーツのリスト
    dimensions: {
      width: number;
      height: number;
    };
  };
}
```

**例**
```json
{
  "success": true,
  "data": {
    "image": "data:image/png;base64,...",
    "format": "png",
    "applied_parts": ["left_eye", "right_eye", "nose"],
    "dimensions": { "width": 512, "height": 512 }
  }
}
```

**エラーケース**
| エラーコード | 条件 | メッセージ |
|-------------|------|-----------|
| VALIDATION_ERROR | partsが無効 | Invalid parts selection format |
| VALIDATION_ERROR | パーツ未選択 | At least one part must be selected |
| FACE_NOT_DETECTED | いずれかの画像で顔未検出 | No face detected in {current/ideal} image |
| PROCESSING_ERROR | 処理失敗 | Failed to generate blended image |

**処理フロー**
1. 両画像から顔のランドマークを検出（MediaPipe Face Mesh）
2. 理想の顔から指定パーツのマスクを生成
3. パーツの位置・角度・サイズを現在の顔に合わせて変換（アフィン変換）
4. 色調を補正（ヒストグラムマッチング）
5. Poisson Blendingでシームレスに合成
6. 合成画像を返却

**対応パーツ一覧**
| パーツID | 名称 | 対象ランドマークインデックス |
|----------|------|------------------------------|
| left_eye | 左目 | 左目・上下まぶた周辺 |
| right_eye | 右目 | 右目・上下まぶた周辺 |
| left_eyebrow | 左眉 | 左眉毛領域 |
| right_eyebrow | 右眉 | 右眉毛領域 |
| nose | 鼻 | 鼻全体（鼻筋・鼻先・小鼻） |
| lips | 口 | 唇・口周り |

**品質要件**
- 処理時間: 10秒以内
- パーツ境界の自然な馴染み
- 色調の不自然な浮きなし

**MVP制約**
- 正面顔のみ対応
- パーツのON/OFFのみ（ブレンド率調整は将来対応）

---

## 3. 画面仕様

### 3.1 画面一覧

| 画面ID | パス | 画面名 | 認証 |
|--------|------|--------|------|
| SCR-001 | `/` | ランディングページ | 不要 |
| SCR-002 | `/simulate` | シミュレーション作成画面 | 不要 |
| SCR-003 | `/simulate/result` | シミュレーション結果画面 | 不要 |
| SCR-004 | `/s/{token}` | 共有閲覧画面 | 不要 |
| SCR-005 | `/login` | ログイン画面 | 不要 |
| SCR-006 | `/mypage` | マイページ | 必要 |
| SCR-007 | `/terms` | 利用規約画面 | 不要 |
| SCR-008 | `/privacy` | プライバシーポリシー画面 | 不要 |

---

### 3.2 SCR-001: ランディングページ

#### ワイヤーフレーム
```
+----------------------------------------------------------+
|  [Logo: Cao]                    [ログイン] [今すぐ試す]  |
+----------------------------------------------------------+
|                                                          |
|                    ┌─────────────────┐                   |
|                    │                 │                   |
|                    │   Hero Image    │                   |
|                    │   (Animation)   │                   |
|                    │                 │                   |
|                    └─────────────────┘                   |
|                                                          |
|         理想の自分を、AIでシミュレーション。             |
|                                                          |
|              [  今すぐ無料で試す  ]                      |
|                                                          |
+----------------------------------------------------------+
|                    使い方                                |
|                                                          |
|   ┌──────┐      ┌──────┐      ┌──────┐                  |
|   │ 1.   │      │ 2.   │      │ 3.   │                  |
|   │Upload│  →   │Upload│  →   │Result│                  |
|   │現在顔│      │理想顔│      │確認  │                  |
|   └──────┘      └──────┘      └──────┘                  |
|                                                          |
+----------------------------------------------------------+
|                 特徴                                     |
|   - 段階的な変化を確認できる                            |
|   - スライダーで自由に調整                              |
|   - 施術者と画像で共有可能                              |
+----------------------------------------------------------+
|  [Footer: 利用規約 | プライバシーポリシー | お問い合わせ]|
+----------------------------------------------------------+
```

#### インタラクション
| 要素 | アクション | 結果 |
|------|-----------|------|
| 「今すぐ試す」ボタン | クリック | `/simulate` へ遷移 |
| 「ログイン」リンク | クリック | `/login` へ遷移 |
| 利用規約リンク | クリック | `/terms` へ遷移 |

---

### 3.3 SCR-002: シミュレーション作成画面

#### ワイヤーフレーム
```
+----------------------------------------------------------+
|  [Logo: Cao]                    [ログイン] [マイページ]  |
+----------------------------------------------------------+
|                                                          |
|    シミュレーション作成                                  |
|                                                          |
|    ┌────────────────────┐  ┌────────────────────┐       |
|    │                    │  │                    │       |
|    │   現在の顔         │  │   理想の顔         │       |
|    │                    │  │                    │       |
|    │  ┌──────────────┐  │  │  ┌──────────────┐  │       |
|    │  │              │  │  │  │              │  │       |
|    │  │   [Preview]  │  │  │  │   [Preview]  │  │       |
|    │  │   or         │  │  │  │   or         │  │       |
|    │  │   Drop here  │  │  │  │   Drop here  │  │       |
|    │  │              │  │  │  │              │  │       |
|    │  └──────────────┘  │  │  └──────────────┘  │       |
|    │                    │  │                    │       |
|    │  [画像を選択]      │  │  [画像を選択]      │       |
|    │  または            │  │  または            │       |
|    │  [×] 削除          │  │  [×] 削除          │       |
|    └────────────────────┘  └────────────────────┘       |
|                                                          |
|    ┌──────────────────────────────────────────────┐     |
|    │        [  シミュレーションを生成  ]          │     |
|    └──────────────────────────────────────────────┘     |
|                                                          |
|    ※ 顔写真は正面を向いた明るい写真をお使いください     |
|                                                          |
+----------------------------------------------------------+
```

#### 状態遷移
```
                    ┌─────────────┐
                    │   初期状態   │
                    │ (両方未選択) │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              │              ▼
    ┌───────────────┐      │      ┌───────────────┐
    │ 現在顔のみ    │      │      │ 理想顔のみ    │
    │ アップロード済│      │      │ アップロード済│
    └───────┬───────┘      │      └───────┬───────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    ┌─────────────┐
                    │  両方選択済  │
                    │ (生成可能)  │
                    └──────┬──────┘
                           │ 生成ボタンクリック
                           ▼
                    ┌─────────────┐
                    │   処理中    │
                    │ (ローディング)│
                    └──────┬──────┘
                           │ 完了
                           ▼
                    ┌─────────────┐
                    │  結果画面へ  │
                    │  遷移       │
                    └─────────────┘
```

#### バリデーションルール

| フィールド | ルール | エラーメッセージ |
|-----------|--------|-----------------|
| 画像形式 | JPEG, PNGのみ | JPEG、PNG形式の画像をアップロードしてください |
| ファイルサイズ | 10MB以下 | 画像サイズは10MB以下にしてください |
| 顔検出 | 顔が1つ検出される | 顔を検出できませんでした。正面を向いた明るい写真をお使いください |
| 複数顔 | 顔が2つ以上 | 複数の顔が検出されました。1人のみ写った写真をお使いください |

#### 利用規約同意モーダル
```
+------------------------------------------+
|         利用規約への同意                 |
+------------------------------------------+
|                                          |
|  サービスをご利用いただく前に、          |
|  利用規約をご確認ください。              |
|                                          |
|  ☐ 利用規約に同意する                   |
|                                          |
|  [利用規約を読む]                        |
|                                          |
|  ┌────────────────────────────────┐     |
|  │        [  同意して続ける  ]     │     |
|  └────────────────────────────────┘     |
|                                          |
+------------------------------------------+
```

---

### 3.4 SCR-003: シミュレーション結果画面

#### ワイヤーフレーム（全体モード）
```
+----------------------------------------------------------+
|  [Logo: Cao]                    [ログイン] [マイページ]  |
+----------------------------------------------------------+
|                                                          |
|    シミュレーション結果                                  |
|                                                          |
|    [全体] [パーツ別適用]                                 |
|                                                          |
|    ┌────────────────────────────────────────────┐       |
|    │                                            │       |
|    │              [Result Image]                │       |
|    │           （ブラーなしで表示）              │       |
|    │                                            │       |
|    └────────────────────────────────────────────┘       |
|                                                          |
|    [現在]  [理想]   ← 切り替えボタン                     |
|                                                          |
|    ┌──────────────┐  ┌──────────────┐                   |
|    │    [保存]    │  │  [共有URL]   │                   |
|    └──────────────┘  └──────────────┘                   |
|                                                          |
+----------------------------------------------------------+
```

#### ワイヤーフレーム（パーツ別適用モード - 未認証）
```
+----------------------------------------------------------+
|  [Logo: Cao]                    [ログイン] [マイページ]  |
+----------------------------------------------------------+
|                                                          |
|    シミュレーション結果                                  |
|                                                          |
|    [全体] [パーツ別適用]                                 |
|                                                          |
|    ┌────────────────────────────────────────────┐       |
|    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│       |
|    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│       |
|    │░░░░░░ [ブラー表示された結果画像] ░░░░░░░░│       |
|    │░░░░░░░░░ タップしてログイン ░░░░░░░░░░░░│       |
|    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│       |
|    └────────────────────────────────────────────┘       |
|                                                          |
|    [現在]  [適用後]   ← 切り替えボタン                   |
|                                                          |
|    パーツ選択: [目] [眉] [鼻] [口]                       |
|                                                          |
|    [適用]                                                |
|                                                          |
+----------------------------------------------------------+
```

#### ワイヤーフレーム（パーツ別適用モード - 認証済み）
```
+----------------------------------------------------------+
|  [Logo: Cao]                    [ユーザー名] [マイページ] |
+----------------------------------------------------------+
|                                                          |
|    シミュレーション結果                                  |
|                                                          |
|    [全体] [パーツ別適用]                                 |
|                                                          |
|    ┌────────────────────────────────────────────┐       |
|    │                                            │       |
|    │              [Result Image]                │       |
|    │           （ブラーなしで表示）              │       |
|    │                                            │       |
|    └────────────────────────────────────────────┘       |
|                                                          |
|    [現在]  [適用後]   ← 切り替えボタン                   |
|                                                          |
|    パーツ選択: [目] [眉] [鼻] [口]                       |
|                                                          |
|    [適用]                                                |
|                                                          |
|    ┌──────────────┐  ┌──────────────┐                   |
|    │    [保存]    │  │  [共有URL]   │                   |
|    └──────────────┘  └──────────────┘                   |
|                                                          |
+----------------------------------------------------------+
```

#### インタラクション
| 要素 | アクション | 条件 | 結果 |
|------|-----------|------|------|
| 全体タブ | クリック | - | 全体シミュレーション結果を表示 |
| パーツ別適用タブ | クリック | - | パーツ別適用モードを表示 |
| 現在/理想ボタン | クリック | 全体モード | 対応する画像を表示 |
| 現在/適用後ボタン | クリック | パーツ別モード | 対応する画像を表示 |
| パーツ別結果画像 | タップ | 未認証・ブラー表示中 | パーツ別ログイン誘導モーダル表示 |
| 保存ボタン | クリック | 未認証 | ログイン誘導モーダル表示 |
| 保存ボタン | クリック | 認証済 | シミュレーション保存 |
| 共有URLボタン | クリック | 未認証 | ログイン誘導モーダル表示 |
| 共有URLボタン | クリック | 認証済・未保存 | 保存後に共有URL生成 |
| 共有URLボタン | クリック | 認証済・保存済 | 共有URL生成 |
| 新規作成ボタン | クリック | - | `/simulate` へ遷移 |

#### パーツ別シミュレーション ログイン誘導モーダル
```
+------------------------------------------+
|    パーツ別の結果を見るには              |
|    ログインが必要です                    |
+------------------------------------------+
|                                          |
|  パーツ別シミュレーションの詳細な結果を  |
|  確認するにはログインしてください。      |
|                                          |
|  ┌────────────────────────────────┐     |
|  │        [  ログインする  ]       │     |
|  └────────────────────────────────┘     |
|                                          |
|  [今はログインしない]                    |
|                                          |
+------------------------------------------+
```

#### 保存/共有 ログイン誘導モーダル
```
+------------------------------------------+
|         保存するにはログインが必要です     |
+------------------------------------------+
|                                          |
|  シミュレーション結果を保存・共有する     |
|  にはログインが必要です。                |
|                                          |
|  ┌────────────────────────────────┐     |
|  │        [  ログインする  ]       │     |
|  └────────────────────────────────┘     |
|                                          |
|  [今はログインしない]                    |
|                                          |
+------------------------------------------+
```

#### 共有URLコピーモーダル
```
+------------------------------------------+
|            共有URLを作成しました          |
+------------------------------------------+
|                                          |
|  以下のURLを施術者に共有してください。    |
|                                          |
|  ┌────────────────────────────────┐     |
|  │ https://cao.app/s/abc123xyz   │     |
|  └────────────────────────────────┘     |
|                                          |
|  ┌────────────────────────────────┐     |
|  │        [  URLをコピー  ]        │     |
|  └────────────────────────────────┘     |
|                                          |
|  [閉じる]                                |
|                                          |
+------------------------------------------+
```

---

### 3.5 SCR-004: 共有閲覧画面

#### ワイヤーフレーム
```
+----------------------------------------------------------+
|  [Logo: Cao]                                              |
+----------------------------------------------------------+
|                                                          |
|    共有されたシミュレーション                             |
|                                                          |
|    ┌────────────────────────────────────────────┐       |
|    │                                            │       |
|    │              [Result Image]                │       |
|    │                                            │       |
|    └────────────────────────────────────────────┘       |
|                                                          |
|    現在 ◀━━━━━━━━━━━━●━━━━━━━━━━━━▶ 理想             |
|            0%   25%   50%   75%   100%                  |
|                                                          |
|    現在の変化度: 50%                                     |
|                                                          |
|    ┌──────────────────────────────────────────────┐     |
|    │      [  自分もシミュレーションを試す  ]       │     |
|    └──────────────────────────────────────────────┘     |
|                                                          |
+----------------------------------------------------------+
```

#### エラー画面（無効なURL）
```
+----------------------------------------------------------+
|  [Logo: Cao]                                              |
+----------------------------------------------------------+
|                                                          |
|                        ⚠                                 |
|                                                          |
|         このURLは無効または期限切れです                   |
|                                                          |
|    ┌──────────────────────────────────────────────┐     |
|    │        [  トップページへ戻る  ]              │     |
|    └──────────────────────────────────────────────┘     |
|                                                          |
+----------------------------------------------------------+
```

---

### 3.6 SCR-005: ログイン画面

#### ワイヤーフレーム
```
+----------------------------------------------------------+
|  [Logo: Cao]                                              |
+----------------------------------------------------------+
|                                                          |
|                    ログイン                              |
|                                                          |
|    ┌────────────────────────────────────────────┐       |
|    │                                            │       |
|    │  メールアドレス                            │       |
|    │  ┌────────────────────────────────────┐   │       |
|    │  │ your@email.com                     │   │       |
|    │  └────────────────────────────────────┘   │       |
|    │                                            │       |
|    │  ┌────────────────────────────────────┐   │       |
|    │  │    [  マジックリンクを送信  ]       │   │       |
|    │  └────────────────────────────────────┘   │       |
|    │                                            │       |
|    │  ─────────── または ───────────           │       |
|    │                                            │       |
|    │  ┌────────────────────────────────────┐   │       |
|    │  │  [G] Googleでログイン              │   │       |
|    │  └────────────────────────────────────┘   │       |
|    │                                            │       |
|    └────────────────────────────────────────────┘       |
|                                                          |
|    ログインすることで、利用規約とプライバシーポリシー    |
|    に同意したものとみなされます。                        |
|                                                          |
+----------------------------------------------------------+
```

#### バリデーションルール
| フィールド | ルール | エラーメッセージ |
|-----------|--------|-----------------|
| メールアドレス | 必須 | メールアドレスを入力してください |
| メールアドレス | 形式 | 有効なメールアドレスを入力してください |

#### マジックリンク送信完了画面
```
+----------------------------------------------------------+
|  [Logo: Cao]                                              |
+----------------------------------------------------------+
|                                                          |
|                    メールを送信しました                   |
|                                                          |
|    ┌────────────────────────────────────────────┐       |
|    │                                            │       |
|    │              ✉                            │       |
|    │                                            │       |
|    │  your@email.com にログインリンクを         │       |
|    │  送信しました。                            │       |
|    │                                            │       |
|    │  メール内のリンクをクリックして            │       |
|    │  ログインしてください。                    │       |
|    │                                            │       |
|    │  ※ メールが届かない場合は迷惑メール        │       |
|    │    フォルダをご確認ください。              │       |
|    │                                            │       |
|    │  [再送信]                                  │       |
|    │                                            │       |
|    └────────────────────────────────────────────┘       |
|                                                          |
+----------------------------------------------------------+
```

---

### 3.7 SCR-006: マイページ

#### ワイヤーフレーム
```
+----------------------------------------------------------+
|  [Logo: Cao]                    [新規作成] [ログアウト]   |
+----------------------------------------------------------+
|                                                          |
|    マイページ                                            |
|                                                          |
|    ┌────────────────────────────────────────────┐       |
|    │  user@email.com                           │       |
|    └────────────────────────────────────────────┘       |
|                                                          |
|    保存済みシミュレーション (3件)                        |
|                                                          |
|    ┌─────────┐  ┌─────────┐  ┌─────────┐               |
|    │         │  │         │  │         │               |
|    │ [Thumb] │  │ [Thumb] │  │ [Thumb] │               |
|    │         │  │         │  │         │               |
|    ├─────────┤  ├─────────┤  ├─────────┤               |
|    │2025/1/23│  │2025/1/22│  │2025/1/20│               |
|    │ [共有]  │  │ [共有]  │  │ [共有]  │               |
|    │ [削除]  │  │ [削除]  │  │ [削除]  │               |
|    └─────────┘  └─────────┘  └─────────┘               |
|                                                          |
|    [さらに読み込む]                                      |
|                                                          |
+----------------------------------------------------------+
```

#### インタラクション
| 要素 | アクション | 結果 |
|------|-----------|------|
| サムネイル | クリック | 結果詳細画面へ遷移 |
| 共有ボタン | クリック | 共有URLコピーモーダル表示 |
| 削除ボタン | クリック | 削除確認モーダル表示 |
| 新規作成ボタン | クリック | `/simulate` へ遷移 |
| ログアウトボタン | クリック | ログアウト後トップへ遷移 |

#### 削除確認モーダル
```
+------------------------------------------+
|          シミュレーションを削除           |
+------------------------------------------+
|                                          |
|  このシミュレーションを削除しますか？    |
|  この操作は取り消せません。              |
|                                          |
|  ┌──────────────┐  ┌──────────────┐     |
|  │  [キャンセル] │  │    [削除]    │     |
|  └──────────────┘  └──────────────┘     |
|                                          |
+------------------------------------------+
```

---

## 4. データモデル

### 4.1 ER図
```
┌─────────────────┐       ┌─────────────────────────────────┐
│   auth.users    │       │           profiles              │
│   (Supabase)    │       │                                 │
├─────────────────┤       ├─────────────────────────────────┤
│ id (PK)         │◄──────│ id (PK, FK)                     │
│ email           │       │ display_name                    │
│ created_at      │       │ avatar_url                      │
│ ...             │       │ created_at                      │
└─────────────────┘       │ updated_at                      │
        │                 └─────────────────────────────────┘
        │
        │ 1:N
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      simulations                            │
├─────────────────────────────────────────────────────────────┤
│ id (PK) UUID                                                │
│ user_id (FK) UUID                                           │
│ current_image_path TEXT                                     │
│ ideal_image_path TEXT                                       │
│ result_images JSONB                                         │
│ settings JSONB                                              │
│ share_token TEXT (UNIQUE, nullable)                         │
│ is_public BOOLEAN                                           │
│ created_at TIMESTAMPTZ                                      │
│ updated_at TIMESTAMPTZ                                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 テーブル定義

#### profiles テーブル
```sql
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- RLS Policy
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
  ON profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE
  USING (auth.uid() = id);

-- Trigger for auto-create profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO profiles (id, display_name)
  VALUES (NEW.id, NEW.email);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION handle_new_user();
```

#### simulations テーブル
```sql
CREATE TABLE simulations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  current_image_path TEXT NOT NULL,
  ideal_image_path TEXT NOT NULL,
  result_images JSONB NOT NULL DEFAULT '[]',
  settings JSONB NOT NULL DEFAULT '{}',
  share_token TEXT UNIQUE,
  is_public BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Indexes
CREATE INDEX idx_simulations_user_id ON simulations(user_id);
CREATE INDEX idx_simulations_share_token ON simulations(share_token) WHERE share_token IS NOT NULL;
CREATE INDEX idx_simulations_created_at ON simulations(created_at DESC);

-- RLS Policy
ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own simulations"
  ON simulations FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own simulations"
  ON simulations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own simulations"
  ON simulations FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own simulations"
  ON simulations FOR DELETE
  USING (auth.uid() = user_id);

CREATE POLICY "Anyone can view public simulations"
  ON simulations FOR SELECT
  USING (is_public = TRUE AND share_token IS NOT NULL);

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at
  BEFORE UPDATE ON simulations
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();
```

### 4.3 JSONBスキーマ

#### result_images
```typescript
type ResultImages = Array<{
  progress: number;    // 0.0 - 1.0
  path: string;        // Storage path
}>;

// 例
[
  { "progress": 0.0, "path": "results/{user_id}/{sim_id}/0.png" },
  { "progress": 0.25, "path": "results/{user_id}/{sim_id}/25.png" },
  { "progress": 0.5, "path": "results/{user_id}/{sim_id}/50.png" },
  { "progress": 0.75, "path": "results/{user_id}/{sim_id}/75.png" },
  { "progress": 1.0, "path": "results/{user_id}/{sim_id}/100.png" }
]
```

#### settings
```typescript
interface SimulationSettings {
  selected_progress?: number;  // ユーザーが選んだ変化度
  notes?: string;              // メモ（将来用）
}

// 例
{
  "selected_progress": 0.5,
  "notes": null
}
```

### 4.4 Storage バケット構成

```
cao-storage/
├── uploads/                    # アップロード画像（認証必須）
│   └── {user_id}/
│       ├── current/
│       │   └── {filename}.{ext}
│       └── ideal/
│           └── {filename}.{ext}
│
└── results/                    # 生成結果（認証必須、共有時は公開）
    └── {user_id}/
        └── {simulation_id}/
            ├── 0.png
            ├── 25.png
            ├── 50.png
            ├── 75.png
            └── 100.png
```

#### Storage Policy
```sql
-- uploads バケット: 本人のみアクセス可
CREATE POLICY "Users can upload own images"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'uploads' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Users can view own uploads"
  ON storage.objects FOR SELECT
  USING (
    bucket_id = 'uploads' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

-- results バケット: 本人または公開シミュレーションの場合
CREATE POLICY "Users can view own results"
  ON storage.objects FOR SELECT
  USING (
    bucket_id = 'results' AND
    (storage.foldername(name))[1] = auth.uid()::text
  );

CREATE POLICY "Anyone can view public results"
  ON storage.objects FOR SELECT
  USING (
    bucket_id = 'results' AND
    EXISTS (
      SELECT 1 FROM simulations
      WHERE is_public = TRUE
      AND id::text = (storage.foldername(name))[2]
    )
  );
```

---

## 5. 非機能要件

### 5.1 パフォーマンス要件

| 項目 | 要件 | 測定方法 |
|------|------|---------|
| ページロード時間 | LCP < 2.5秒 | Lighthouse |
| API レスポンス（分析） | p95 < 3秒 | サーバーログ |
| API レスポンス（モーフィング） | p95 < 10秒 | サーバーログ |
| Time to Interactive | < 3秒 | Lighthouse |
| First Input Delay | < 100ms | Web Vitals |

### 5.2 スケーラビリティ要件

| 項目 | MVP要件 | 将来目標 |
|------|--------|---------|
| 同時接続ユーザー数 | 100 | 10,000 |
| 1日あたりシミュレーション数 | 1,000 | 100,000 |
| 画像ストレージ | 100GB | 10TB |

### 5.3 可用性要件

| 項目 | 要件 |
|------|------|
| サービス稼働率 | 99.5% |
| 計画メンテナンス | 月1回、深夜帯 |
| 障害復旧時間（RTO） | 4時間以内 |
| データ復旧ポイント（RPO） | 24時間以内 |

### 5.4 セキュリティ要件

#### 認証・認可
| 要件ID | 要件 | 実装方法 |
|--------|------|---------|
| SEC-AUTH-001 | パスワードレス認証 | Supabase Auth Magic Link |
| SEC-AUTH-002 | OAuth認証 | Google認証 |
| SEC-AUTH-003 | セッション管理 | JWT (有効期限7日) |
| SEC-AUTH-004 | CSRF対策 | SameSite Cookie |

#### データ保護
| 要件ID | 要件 | 実装方法 |
|--------|------|---------|
| SEC-DATA-001 | 通信暗号化 | TLS 1.3 |
| SEC-DATA-002 | データベース暗号化 | Supabase暗号化 |
| SEC-DATA-003 | 画像プライバシー | 署名付きURL、RLS |
| SEC-DATA-004 | 個人情報保護 | メールアドレスのみ保存 |

#### 入力検証
| 要件ID | 要件 | 実装方法 |
|--------|------|---------|
| SEC-INPUT-001 | ファイル形式検証 | Magic bytes検証 |
| SEC-INPUT-002 | ファイルサイズ制限 | 10MB上限 |
| SEC-INPUT-003 | XSS対策 | React自動エスケープ |
| SEC-INPUT-004 | SQLインジェクション対策 | Supabase クライアント |

#### レート制限
| ユーザー種別 | 制限 |
|-------------|------|
| 未認証 | 10 requests/minute |
| 認証済み | 30 requests/minute |

### 5.5 アクセシビリティ要件

| 要件ID | 要件 | 基準 |
|--------|------|------|
| A11Y-001 | キーボード操作対応 | 全機能がキーボードで操作可能 |
| A11Y-002 | スクリーンリーダー対応 | 適切なaria-label設定 |
| A11Y-003 | コントラスト比 | WCAG 2.1 AA準拠 (4.5:1以上) |
| A11Y-004 | フォーカス可視化 | フォーカスリングの視認性確保 |

### 5.6 ブラウザ対応

| ブラウザ | バージョン | 対応レベル |
|---------|-----------|-----------|
| Chrome | 最新2バージョン | フルサポート |
| Safari | 最新2バージョン | フルサポート |
| Firefox | 最新2バージョン | フルサポート |
| Edge | 最新2バージョン | フルサポート |
| Safari (iOS) | 最新2バージョン | フルサポート |
| Chrome (Android) | 最新2バージョン | フルサポート |

### 5.7 国際化要件

| 項目 | MVP対応 | 備考 |
|------|--------|------|
| 対応言語 | 日本語のみ | 将来的に英語対応予定 |
| 日時形式 | JST | `YYYY/MM/DD HH:mm` |
| 通貨 | N/A | MVPでは課金なし |

---

## 6. エラーハンドリング

### 6.1 フロントエンド エラーメッセージ一覧

| エラーコード | 日本語メッセージ | 表示位置 |
|-------------|-----------------|---------|
| VALIDATION_ERROR | 入力内容を確認してください | フォーム下部 |
| FACE_NOT_DETECTED | 顔を検出できませんでした。正面を向いた明るい写真をお使いください | アップロードエリア |
| MULTIPLE_FACES | 複数の顔が検出されました。1人のみ写った写真をお使いください | アップロードエリア |
| IMAGE_TOO_LARGE | 画像サイズは10MB以下にしてください | アップロードエリア |
| INVALID_IMAGE_FORMAT | JPEG、PNG形式の画像をアップロードしてください | アップロードエリア |
| PROCESSING_ERROR | 処理中にエラーが発生しました。再度お試しください | トースト通知 |
| RATE_LIMITED | リクエストが多すぎます。しばらく待ってから再度お試しください | トースト通知 |
| UNAUTHORIZED | ログインが必要です | モーダル |
| NOT_FOUND | 指定されたデータが見つかりません | ページ内 |
| NETWORK_ERROR | 通信エラーが発生しました。ネットワーク接続を確認してください | トースト通知 |

### 6.2 リトライ戦略

| エラー種別 | 自動リトライ | リトライ回数 | 間隔 |
|-----------|-------------|-------------|------|
| ネットワークエラー | Yes | 3回 | 指数バックオフ (1s, 2s, 4s) |
| 5xxエラー | Yes | 2回 | 指数バックオフ (2s, 4s) |
| 429エラー | Yes | 1回 | Retry-Afterヘッダーに従う |
| 4xxエラー | No | - | - |

---

## 7. テスト要件

### 7.1 テストカバレッジ目標

| テスト種別 | カバレッジ目標 | 対象 |
|-----------|---------------|------|
| 単体テスト | 80%以上 | ビジネスロジック、ユーティリティ |
| 結合テスト | 70%以上 | APIエンドポイント |
| E2Eテスト | 主要ユースケース100% | UC-001〜UC-010 |

### 7.2 E2Eテストシナリオ

| シナリオID | ユースケース | 説明 |
|-----------|-------------|------|
| E2E-001 | UC-001 | 利用規約同意フロー |
| E2E-002 | UC-002, UC-003 | 画像アップロード（成功） |
| E2E-003 | UC-002 | 画像アップロード（エラー: 顔未検出） |
| E2E-004 | UC-004, UC-005 | シミュレーション生成・確認 |
| E2E-005 | UC-009 | ユーザー登録（メール） |
| E2E-006 | UC-009 | ユーザー登録（Google） |
| E2E-007 | UC-006 | シミュレーション保存 |
| E2E-008 | UC-007 | 共有URL生成 |
| E2E-009 | UC-008 | 共有URL閲覧 |
| E2E-010 | UC-010 | 履歴確認 |
| E2E-011 | UC-011 | パーツ別ブレンド生成（成功・認証済み） |
| E2E-012 | UC-011 | パーツ別ブレンド（エラー: 顔未検出） |
| E2E-013 | UC-011 | パーツ別ブレンド生成（未認証・ブラー表示） |
| E2E-014 | UC-012 | ブラー画像タップからログイン・結果閲覧 |

---

## 変更履歴

| バージョン | 日付 | 変更内容 | 担当 |
|------------|------|----------|------|
| 1.0.0 | 2025-01-23 | 初版作成 | Spec Agent |
| 1.1.0 | 2025-01-25 | パーツ別ブレンドAPI（2.3.11）追加、E2E-011/012追加 | Spec Agent |
| 1.2.0 | 2025-01-27 | パーツ別結果のブラー表示仕様追加（SCR-003更新）、E2E-013/014追加 | Spec Agent |
