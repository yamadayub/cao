/**
 * 非同期生成ジョブAPI
 *
 * 顔シミュレーション生成を非同期ジョブとして実行するAPI。
 * 長時間の処理をバックグラウンドで実行し、ポーリングで結果を取得する。
 *
 * @example
 * ```typescript
 * // ジョブ作成
 * const job = await createGenerationJob({
 *   base_image: base64Current,
 *   target_image: base64Ideal,
 *   mode: 'morph',
 *   strength: 0.5,
 * });
 *
 * // 完了まで待機
 * const result = await waitForJobCompletion(job.job_id);
 * if (result.status === 'succeeded') {
 *   const imgSrc = result.result_image_url;
 * }
 * ```
 */

import { apiGet, apiPost, ApiError } from './client';
import {
  CreateGenerationJobRequest,
  GenerationJobStatus,
  GenerationResultData,
  JobStatus,
} from './types';

/**
 * デフォルトのポーリング間隔（ミリ秒）
 */
const DEFAULT_POLL_INTERVAL = 2000;

/**
 * デフォルトの最大待機時間（ミリ秒）
 */
const DEFAULT_MAX_WAIT_TIME = 300000; // 5分

/**
 * 生成ジョブを作成
 *
 * @param request - ジョブ作成リクエスト
 * @param authToken - 認証トークン（オプション）
 * @returns 作成されたジョブのステータス
 *
 * @example
 * ```typescript
 * const job = await createGenerationJob({
 *   base_image: base64Data,
 *   target_image: base64Ideal,
 *   mode: 'parts',
 *   parts: ['eyes', 'nose', 'lips'],
 *   strength: 0.7,
 * });
 * console.log('Job created:', job.job_id);
 * ```
 */
export async function createGenerationJob(
  request: CreateGenerationJobRequest,
  authToken?: string
): Promise<GenerationJobStatus> {
  const options = authToken ? { authToken } : undefined;
  return apiPost<GenerationJobStatus>(
    '/api/v1/simulations/generate',
    request,
    options
  );
}

/**
 * ジョブのステータスを取得
 *
 * @param jobId - ジョブID
 * @returns ジョブステータス
 */
export async function getJobStatus(jobId: string): Promise<GenerationJobStatus> {
  return apiGet<GenerationJobStatus>(`/api/v1/simulations/generate/${jobId}`);
}

/**
 * ジョブの結果を取得
 *
 * @param jobId - ジョブID
 * @returns 生成結果（Base64画像データ含む）
 * @throws ApiError - ジョブが完了していない場合
 */
export async function getJobResult(jobId: string): Promise<GenerationResultData> {
  return apiGet<GenerationResultData>(`/api/v1/simulations/generate/${jobId}/result`);
}

/**
 * ジョブが完了するまで待機
 *
 * @param jobId - ジョブID
 * @param options - 待機オプション
 * @returns 完了時のジョブステータス
 * @throws Error - タイムアウトまたはジョブ失敗時
 *
 * @example
 * ```typescript
 * const result = await waitForJobCompletion(job.job_id, {
 *   pollInterval: 1000,
 *   maxWaitTime: 60000,
 *   onProgress: (status) => {
 *     console.log(`Progress: ${status.progress}%`);
 *   },
 * });
 * ```
 */
export async function waitForJobCompletion(
  jobId: string,
  options: {
    pollInterval?: number;
    maxWaitTime?: number;
    onProgress?: (status: GenerationJobStatus) => void;
  } = {}
): Promise<GenerationJobStatus> {
  const {
    pollInterval = DEFAULT_POLL_INTERVAL,
    maxWaitTime = DEFAULT_MAX_WAIT_TIME,
    onProgress,
  } = options;

  const startTime = Date.now();

  while (true) {
    const status = await getJobStatus(jobId);

    // プログレスコールバック
    if (onProgress) {
      onProgress(status);
    }

    // 終了状態をチェック
    if (status.status === 'succeeded') {
      return status;
    }

    if (status.status === 'failed') {
      throw new Error(status.error || 'Job failed');
    }

    // タイムアウトチェック
    if (Date.now() - startTime > maxWaitTime) {
      throw new Error('Job timed out');
    }

    // 次のポーリングまで待機
    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }
}

/**
 * ジョブを作成し完了まで待機するユーティリティ
 *
 * createGenerationJob + waitForJobCompletion を組み合わせた便利関数。
 *
 * @param request - ジョブ作成リクエスト
 * @param options - 待機オプション
 * @returns 生成結果
 *
 * @example
 * ```typescript
 * try {
 *   const result = await generateAndWait({
 *     base_image: base64Current,
 *     target_image: base64Ideal,
 *     mode: 'morph',
 *     strength: 0.5,
 *   }, {
 *     onProgress: (status) => setProgress(status.progress),
 *   });
 *   setResultImage(result.result_image_url);
 * } catch (error) {
 *   console.error('Generation failed:', error);
 * }
 * ```
 */
export async function generateAndWait(
  request: CreateGenerationJobRequest,
  options: {
    authToken?: string;
    pollInterval?: number;
    maxWaitTime?: number;
    onProgress?: (status: GenerationJobStatus) => void;
  } = {}
): Promise<GenerationJobStatus> {
  const { authToken, ...waitOptions } = options;

  // ジョブ作成
  const job = await createGenerationJob(request, authToken);

  // 初期プログレスコールバック
  if (waitOptions.onProgress) {
    waitOptions.onProgress(job);
  }

  // 完了まで待機
  return waitForJobCompletion(job.job_id, waitOptions);
}

/**
 * Base64画像をFileに変換するユーティリティ
 *
 * @param base64 - Base64エンコードされた画像データ
 * @param filename - ファイル名
 * @returns Fileオブジェクト
 */
export function base64ToFile(base64: string, filename: string = 'image.png'): File {
  // Data URL形式の場合、プレフィックスを除去
  const base64Data = base64.startsWith('data:')
    ? base64.split(',')[1]
    : base64;

  const byteCharacters = atob(base64Data);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);

  return new File([byteArray], filename, { type: 'image/png' });
}

/**
 * FileをBase64に変換するユーティリティ
 *
 * @param file - Fileオブジェクト
 * @returns Base64エンコードされた文字列（Data URLプレフィックス付き）
 */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
