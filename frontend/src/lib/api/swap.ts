/**
 * Face Swap API
 *
 * Replicate APIを使用したFace Swap機能とパーツ合成を提供
 *
 * @example
 * ```typescript
 * // 1. Face Swapを実行
 * const job = await generateSwap({
 *   current_image: base64Current,
 *   ideal_image: base64Ideal,
 * });
 *
 * // 2. 結果を取得（ポーリング）
 * const result = await getSwapResult(job.job_id);
 * if (result.status === 'completed') {
 *   // 3. パーツを適用
 *   const final = await applySwapParts({
 *     current_image: base64Current,
 *     swapped_image: result.swapped_image,
 *     parts: { eyes: 0.8, nose: 1.0, lips: 0.5 },
 *   });
 * }
 * ```
 */

import { apiPost, apiGet } from './client';
import {
  SwapGenerateRequest,
  SwapGenerateData,
  SwapResultData,
  SwapPartsRequest,
  SwapPartsData,
  SwapPreviewAllRequest,
  SwapPreviewAllData,
  SwapJobStatus,
  SwapPartsIntensity,
} from './types';

/**
 * デフォルトのポーリング間隔（ミリ秒）
 */
const DEFAULT_POLL_INTERVAL = 2000;

/**
 * デフォルトの最大待機時間（ミリ秒）
 */
const DEFAULT_MAX_WAIT_TIME = 120000; // 2分

/**
 * Face Swapジョブを開始
 *
 * @param request - 現在の画像と理想の画像（Base64）
 * @returns ジョブIDとステータス
 *
 * @example
 * ```typescript
 * const job = await generateSwap({
 *   current_image: base64Current,
 *   ideal_image: base64Ideal,
 * });
 * console.log('Job started:', job.job_id);
 * ```
 */
export async function generateSwap(
  request: SwapGenerateRequest
): Promise<SwapGenerateData> {
  return apiPost<SwapGenerateData>('/api/v1/swap/generate', request, {
    timeout: 60000, // 60秒
  });
}

/**
 * Face Swapの結果を取得
 *
 * @param jobId - ジョブID
 * @returns ジョブステータスと結果画像（完了時）
 *
 * @example
 * ```typescript
 * const result = await getSwapResult(job.job_id);
 * if (result.status === 'completed') {
 *   const imgSrc = `data:image/png;base64,${result.swapped_image}`;
 * }
 * ```
 */
export async function getSwapResult(jobId: string): Promise<SwapResultData> {
  return apiGet<SwapResultData>(`/api/v1/swap/generate/${jobId}`);
}

/**
 * Face Swap完了まで待機
 *
 * @param jobId - ジョブID
 * @param options - ポーリングオプション
 * @returns 完了時の結果データ
 * @throws Error - タイムアウトまたはジョブ失敗時
 *
 * @example
 * ```typescript
 * const result = await waitForSwapCompletion(job.job_id, {
 *   onProgress: (status) => console.log('Status:', status),
 * });
 * ```
 */
export async function waitForSwapCompletion(
  jobId: string,
  options: {
    pollInterval?: number;
    maxWaitTime?: number;
    onProgress?: (status: SwapJobStatus) => void;
  } = {}
): Promise<SwapResultData> {
  const {
    pollInterval = DEFAULT_POLL_INTERVAL,
    maxWaitTime = DEFAULT_MAX_WAIT_TIME,
    onProgress,
  } = options;

  const startTime = Date.now();

  while (true) {
    const result = await getSwapResult(jobId);

    // プログレスコールバック
    if (onProgress) {
      onProgress(result.status);
    }

    // 終了状態をチェック
    if (result.status === 'completed') {
      return result;
    }

    if (result.status === 'failed') {
      throw new Error(result.error || 'Face swap job failed');
    }

    // タイムアウトチェック
    if (Date.now() - startTime > maxWaitTime) {
      throw new Error('Face swap job timed out');
    }

    // 次のポーリングまで待機
    await new Promise((resolve) => setTimeout(resolve, pollInterval));
  }
}

/**
 * Face Swapを実行し完了まで待機
 *
 * generateSwap + waitForSwapCompletion を組み合わせた便利関数
 *
 * @param request - リクエストデータ
 * @param options - ポーリングオプション
 * @returns 完了時の結果データ（swapped_image含む）
 *
 * @example
 * ```typescript
 * const result = await swapAndWait({
 *   current_image: base64Current,
 *   ideal_image: base64Ideal,
 * }, {
 *   onProgress: (status) => setStatus(status),
 * });
 * setSwappedImage(result.swapped_image);
 * ```
 */
export async function swapAndWait(
  request: SwapGenerateRequest,
  options: {
    pollInterval?: number;
    maxWaitTime?: number;
    onProgress?: (status: SwapJobStatus) => void;
  } = {}
): Promise<SwapResultData> {
  // ジョブ開始
  const job = await generateSwap(request);

  // 初期プログレスコールバック
  if (options.onProgress) {
    options.onProgress(job.status);
  }

  // 完了まで待機
  return waitForSwapCompletion(job.job_id, options);
}

/**
 * パーツ合成を適用
 *
 * Face Swap結果に対してパーツごとの強度を適用
 *
 * @param request - 元画像、スワップ画像、パーツ強度
 * @returns 合成結果画像（Base64）
 *
 * @example
 * ```typescript
 * const result = await applySwapParts({
 *   current_image: base64Current,
 *   swapped_image: base64Swapped,
 *   parts: {
 *     eyes: 1.0,      // 目を100%適用
 *     nose: 0.5,      // 鼻を50%適用
 *     lips: 0.0,      // 唇は適用しない
 *   },
 * });
 * ```
 */
export async function applySwapParts(
  request: SwapPartsRequest
): Promise<SwapPartsData> {
  return apiPost<SwapPartsData>('/api/v1/swap/parts', request, {
    timeout: 60000, // 60秒
  });
}

/**
 * 全パーツプレビューを生成
 *
 * @param request - 元画像、スワップ画像、パーツ強度
 * @returns プレビュー結果画像（Base64）
 *
 * @example
 * ```typescript
 * const result = await previewAllParts({
 *   current_image: base64Current,
 *   swapped_image: base64Swapped,
 *   parts: { eyes: 0.8, nose: 1.0, lips: 0.5 },
 * });
 * setPreviewImage(result.result_image);
 * ```
 */
export async function previewAllParts(
  request: SwapPreviewAllRequest
): Promise<SwapPreviewAllData> {
  return apiPost<SwapPreviewAllData>('/api/v1/swap/preview-all', request, {
    timeout: 60000, // 60秒
  });
}

/**
 * デフォルトのパーツ強度
 */
export const DEFAULT_PARTS_INTENSITY: SwapPartsIntensity = {
  left_eye: 1.0,
  right_eye: 1.0,
  left_eyebrow: 1.0,
  right_eyebrow: 1.0,
  nose: 1.0,
  lips: 1.0,
};

/**
 * パーツ強度をリセット（全て0）
 */
export const ZERO_PARTS_INTENSITY: SwapPartsIntensity = {
  left_eye: 0,
  right_eye: 0,
  left_eyebrow: 0,
  right_eyebrow: 0,
  nose: 0,
  lips: 0,
};
