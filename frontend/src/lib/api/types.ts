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
  | 'INTERNAL_ERROR'
  | 'JOB_FAILED';

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
  swapped_image?: string;  // パーツモード用のスワップ画像（Base64）
  settings?: {
    selected_progress?: number;
    notes?: string;
  };
}

/**
 * 結果画像
 */
export interface ResultImage {
  progress: number;
  image: string;  // Base64またはURL
  url?: string;   // 後方互換性のため（廃止予定）
}

/**
 * シミュレーションデータ
 */
export interface SimulationData {
  id: string;  // UUID
  user_id: string;
  current_image_url: string;
  ideal_image_url: string;
  swapped_image_url: string | null;  // パーツモード用のスワップ画像URL
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
 * パーツ選択（シンプル3択）
 * - eyes: 目と眉毛をセットで適用
 * - nose: 鼻
 * - lips: 唇
 */
export interface PartsSelection {
  eyes: boolean;
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

// =============================================================================
// Generation Jobs (非同期ジョブAPI)
// =============================================================================

/**
 * ジョブ生成モード
 */
export type GenerationMode = 'morph' | 'parts';

/**
 * ジョブステータス
 */
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed';

/**
 * ジョブ作成リクエスト
 */
export interface CreateGenerationJobRequest {
  base_image: string;      // Base64エンコード
  target_image: string;    // Base64エンコード
  mode: GenerationMode;
  parts?: string[];        // mode='parts'の場合必須
  strength?: number;       // 0-1、デフォルト0.5
  seed?: number;
}

/**
 * ジョブステータスデータ
 */
export interface GenerationJobStatus {
  job_id: string;
  status: JobStatus;
  progress: number;        // 0-100
  result_image_url: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export type CreateGenerationJobResponse = SuccessResponse<GenerationJobStatus>;
export type GenerationJobStatusResponse = SuccessResponse<GenerationJobStatus>;

/**
 * ジョブ結果データ
 */
export interface GenerationResultData {
  job_id: string;
  image: string;           // Base64エンコード
  format: 'png';
  mode: GenerationMode;
  strength: number;
}

export type GenerationResultResponse = SuccessResponse<GenerationResultData>;

/**
 * パーツ表示名マッピング
 */
export const PARTS_DISPLAY_NAMES: Record<keyof PartsSelection, string> = {
  eyes: '目',
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
  JOB_FAILED: '生成ジョブが失敗しました。再度お試しください',
};

/**
 * エラーコードからローカライズされたメッセージを取得
 */
export function getErrorMessage(code: ErrorCode): string {
  return ERROR_MESSAGES[code] || ERROR_MESSAGES.INTERNAL_ERROR;
}

// =============================================================================
// Face Swap API
// =============================================================================

/**
 * Face Swap ジョブステータス
 */
export type SwapJobStatus = 'pending' | 'processing' | 'completed' | 'failed';

/**
 * Face Swap 生成リクエスト
 */
export interface SwapGenerateRequest {
  current_image: string;  // Base64エンコード
  ideal_image: string;    // Base64エンコード
}

/**
 * Face Swap 生成レスポンスデータ
 */
export interface SwapGenerateData {
  job_id: string;
  status: SwapJobStatus;
}

export type SwapGenerateResponse = SuccessResponse<SwapGenerateData>;

/**
 * Face Swap 結果データ
 */
export interface SwapResultData {
  status: SwapJobStatus;
  swapped_image: string | null;  // Base64エンコード（完了時）
  error: string | null;
}

export type SwapResultResponse = SuccessResponse<SwapResultData>;

/**
 * パーツ強度指定（0.0-1.0）
 * シンプル3択: eyes（目+眉）, nose, lips
 */
export interface SwapPartsIntensity {
  eyes?: number;    // 目と眉毛をセットで適用
  nose?: number;
  lips?: number;
}

/**
 * Face Swap パーツ合成リクエスト
 */
export interface SwapPartsRequest {
  current_image: string;   // Base64エンコード（オリジナル）
  swapped_image: string;   // Base64エンコード（スワップ結果）
  parts: SwapPartsIntensity;
}

/**
 * Face Swap パーツ合成結果
 */
export interface SwapPartsData {
  result_image: string;  // Base64エンコード
}

export type SwapPartsResponse = SuccessResponse<SwapPartsData>;

/**
 * Face Swap プレビュー全体リクエスト
 */
export interface SwapPreviewAllRequest {
  current_image: string;
  swapped_image: string;
  parts: SwapPartsIntensity;
}

/**
 * Face Swap プレビュー全体結果
 */
export interface SwapPreviewAllData {
  result_image: string;
}

export type SwapPreviewAllResponse = SuccessResponse<SwapPreviewAllData>;

// =============================================================================
// SNS Share API
// =============================================================================

/**
 * シェア画像タイプ
 * - before_after: Before/After比較画像（1200x630px）
 * - result_only: 結果のみ画像（1080x1080px）
 */
export type ShareType = 'before_after' | 'result_only';

/**
 * @deprecated ShareTemplateはShareTypeに置き換えられました
 */
export type ShareTemplate = 'before_after' | 'single' | 'parts_highlight';

/**
 * SNSシェア作成リクエスト
 */
export interface CreateSnsShareRequest {
  source_image: string;       // Base64エンコード
  result_image: string;       // Base64エンコード
  share_type: ShareType;
}

/**
 * SNSシェア作成レスポンスデータ
 */
export interface SnsShareData {
  share_id: string;
  share_url: string;
  share_image_url: string;
  og_image_url: string;
  expires_at: string;
}

export type CreateSnsShareResponse = SuccessResponse<SnsShareData>;

/**
 * SNSシェア取得レスポンスデータ
 */
export interface GetSnsShareData {
  share_id: string;
  share_image_url: string;
  share_type: ShareType;
  created_at: string;
  expires_at: string;
  is_expired: boolean;
}

export type GetSnsShareResponse = SuccessResponse<GetSnsShareData>;
