/**
 * APIクライアント - エクスポート
 *
 * @example
 * ```typescript
 * import { analyzeImage, morphStages, ApiError } from '@/lib/api';
 *
 * async function processImages(current: File, ideal: File) {
 *   try {
 *     // 両方の画像を分析
 *     const [currentAnalysis, idealAnalysis] = await Promise.all([
 *       analyzeImage(current),
 *       analyzeImage(ideal),
 *     ]);
 *
 *     // モーフィング実行
 *     const result = await morphStages(current, ideal);
 *     return result.images;
 *   } catch (error) {
 *     if (error instanceof ApiError) {
 *       console.error(error.localizedMessage);
 *     }
 *     throw error;
 *   }
 * }
 * ```
 */

// クライアント
export {
  API_BASE_URL,
  apiClient,
  apiGet,
  apiPost,
  apiPostFormData,
  apiDelete,
  ApiError,
  NetworkError,
  TimeoutError,
} from './client';
export type { ApiClientOptions } from './client';

// 型定義
export type {
  // 共通
  ResponseMeta,
  SuccessResponse,
  ErrorCode,
  ErrorDetail,
  ErrorResponse,
  ApiResponse,
  // Health
  HealthData,
  HealthResponse,
  // Analyze
  FaceRegion,
  FaceLandmark,
  ImageInfo,
  AnalyzeData,
  AnalyzeResponse,
  // Morph
  ImageDimensions,
  MorphData,
  MorphResponse,
  StageImage,
  StagedMorphData,
  StagedMorphResponse,
  // Simulations
  CreateSimulationRequest,
  ResultImage,
  SimulationData,
  SimulationResponse,
  SimulationSummary,
  Pagination,
  SimulationListData,
  SimulationListResponse,
  DeleteData,
  DeleteResponse,
  ShareData,
  ShareResponse,
  SharedSimulationData,
  SharedSimulationResponse,
  // Generation Jobs
  GenerationMode,
  JobStatus,
  CreateGenerationJobRequest,
  GenerationJobStatus,
  GenerationResultData,
  // Face Swap
  SwapJobStatus,
  SwapGenerateRequest,
  SwapGenerateData,
  SwapResultData,
  SwapPartsRequest,
  SwapPartsData,
  SwapPartsIntensity,
  SwapPreviewAllRequest,
  SwapPreviewAllData,
} from './types';
export { ERROR_MESSAGES, getErrorMessage } from './types';

// API関数
export { checkHealth } from './health';
export { analyzeImage } from './analyze';
export { morphImages, morphStages, toDataUrl, DEFAULT_STAGES } from './morph';
export {
  createSimulation,
  getSimulations,
  getSimulation,
  deleteSimulation,
  createShareUrl,
  getSharedSimulation,
} from './simulations';

// 非同期生成ジョブAPI
export {
  createGenerationJob,
  getJobStatus,
  getJobResult,
  waitForJobCompletion,
  generateAndWait,
  base64ToFile,
  fileToBase64,
} from './generation';

// Face Swap API
export {
  generateSwap,
  getSwapResult,
  swapAndWait,
  applySwapParts,
  previewAllParts,
  DEFAULT_PARTS_INTENSITY,
  ZERO_PARTS_INTENSITY,
} from './swap';
