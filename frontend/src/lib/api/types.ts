/**
 * API型定義
 * functional-spec.mdのスキーマに基づく型定義
 */

// =============================================================================
// 共通型定義
// =============================================================================

/**
 * レスポンスメタ情報
 */
export interface ResponseMeta {
  request_id: string;
  processing_time_ms: number;
}

/**
 * 成功レスポンス
 */
export interface SuccessResponse<T> {
  success: true;
  data: T;
  meta?: ResponseMeta;
}

/**
 * エラーコード
 */
export type ErrorCode =
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

/**
 * エラー詳細
 */
export interface ErrorDetail {
  code: ErrorCode;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * エラーレスポンス
 */
export interface ErrorResponse {
  success: false;
  error: ErrorDetail;
}

/**
 * APIレスポンス（成功またはエラー）
 */
export type ApiResponse<T> = SuccessResponse<T> | ErrorResponse;

// =============================================================================
// Health Check
// =============================================================================

export interface HealthData {
  status: 'ok' | 'degraded';
  version: string;
  timestamp: string;
}

export type HealthResponse = SuccessResponse<HealthData>;

// =============================================================================
// 顔分析 (Analyze)
// =============================================================================

/**
 * 顔領域
 */
export interface FaceRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * 顔のランドマーク
 */
export interface FaceLandmark {
  index: number;
  x: number;  // 正規化座標 (0.0 - 1.0)
  y: number;
  z: number;
}

/**
 * 画像情報
 */
export interface ImageInfo {
  width: number;
  height: number;
  format: 'jpeg' | 'png';
}

/**
 * 顔分析データ
 */
export interface AnalyzeData {
  face_detected: boolean;
  face_count: number;
  face_region: FaceRegion | null;
  landmarks: FaceLandmark[] | null;
  image_info: ImageInfo;
}

export type AnalyzeResponse = SuccessResponse<AnalyzeData>;

// =============================================================================
// モーフィング (Morph)
// =============================================================================

/**
 * 画像サイズ
 */
export interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * 単一モーフィングデータ
 */
export interface MorphData {
  image: string;  // Base64エンコード
  format: 'png';
  progress: number;
  dimensions: ImageDimensions;
}

export type MorphResponse = SuccessResponse<MorphData>;

/**
 * 段階別画像
 */
export interface StageImage {
  progress: number;
  image: string;  // Base64エンコード
}

/**
 * 段階的モーフィングデータ
 */
export interface StagedMorphData {
  images: StageImage[];
  format: 'png';
  dimensions: ImageDimensions;
}

export type StagedMorphResponse = SuccessResponse<StagedMorphData>;

// =============================================================================
// シミュレーション (Simulations)
// =============================================================================

/**
 * シミュレーション作成リクエスト
 */
export interface CreateSimulationRequest {
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

/**
 * 結果画像（URL付き）
 */
export interface ResultImage {
  progress: number;
  url: string;
}

/**
 * シミュレーションデータ
 */
export interface SimulationData {
  id: string;  // UUID
  user_id: string;
  current_image_url: string;
  ideal_image_url: string;
  result_images: ResultImage[];
  settings: Record<string, unknown>;
  share_token: string | null;
  is_public: boolean;
  created_at: string;
  updated_at: string;
}

export type SimulationResponse = SuccessResponse<SimulationData>;

/**
 * シミュレーションサマリー
 */
export interface SimulationSummary {
  id: string;
  thumbnail_url: string;
  created_at: string;
  is_public: boolean;
}

/**
 * ページネーション情報
 */
export interface Pagination {
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

/**
 * シミュレーション一覧データ
 */
export interface SimulationListData {
  simulations: SimulationSummary[];
  pagination: Pagination;
}

export type SimulationListResponse = SuccessResponse<SimulationListData>;

/**
 * 削除レスポンスデータ
 */
export interface DeleteData {
  deleted: true;
  id: string;
}

export type DeleteResponse = SuccessResponse<DeleteData>;

/**
 * 共有レスポンスデータ
 */
export interface ShareData {
  share_token: string;
  share_url: string;
  expires_at: null;
}

export type ShareResponse = SuccessResponse<ShareData>;

/**
 * 共有シミュレーションデータ
 */
export interface SharedSimulationData {
  result_images: ResultImage[];
  created_at: string;
}

export type SharedSimulationResponse = SuccessResponse<SharedSimulationData>;

// =============================================================================
// パーツブレンド (Parts Blend)
// =============================================================================

/**
 * パーツ選択
 */
export interface PartsSelection {
  left_eye: boolean;
  right_eye: boolean;
  left_eyebrow: boolean;
  right_eyebrow: boolean;
  nose: boolean;
  lips: boolean;
}

/**
 * パーツブレンドデータ
 */
export interface PartsBlendData {
  image: string;  // Base64エンコード
  format: 'png';
  applied_parts: string[];
  dimensions: ImageDimensions;
}

export type PartsBlendResponse = SuccessResponse<PartsBlendData>;

/**
 * パーツ表示名マッピング
 */
export const PARTS_DISPLAY_NAMES: Record<keyof PartsSelection, string> = {
  left_eye: '左目',
  right_eye: '右目',
  left_eyebrow: '左眉',
  right_eyebrow: '右眉',
  nose: '鼻',
  lips: '唇',
};

// =============================================================================
// エラーメッセージマッピング
// =============================================================================

/**
 * エラーコードに対応する日本語メッセージ
 */
export const ERROR_MESSAGES: Record<ErrorCode, string> = {
  VALIDATION_ERROR: '入力内容を確認してください',
  FACE_NOT_DETECTED: '顔を検出できませんでした。正面を向いた明るい写真をお使いください',
  MULTIPLE_FACES: '複数の顔が検出されました。1人のみ写った写真をお使いください',
  IMAGE_TOO_LARGE: '画像サイズは10MB以下にしてください',
  INVALID_IMAGE_FORMAT: 'JPEG、PNG形式の画像をアップロードしてください',
  PROCESSING_ERROR: '処理中にエラーが発生しました。再度お試しください',
  RATE_LIMITED: 'リクエストが多すぎます。しばらく待ってから再度お試しください',
  UNAUTHORIZED: 'ログインが必要です',
  NOT_FOUND: '指定されたデータが見つかりません',
  INTERNAL_ERROR: 'サーバーエラーが発生しました。しばらく待ってから再度お試しください',
};

/**
 * エラーコードからローカライズされたメッセージを取得
 */
export function getErrorMessage(code: ErrorCode): string {
  return ERROR_MESSAGES[code] || ERROR_MESSAGES.INTERNAL_ERROR;
}
