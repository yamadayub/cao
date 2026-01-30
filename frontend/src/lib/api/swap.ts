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
  SwapResultData,
  SwapPartsRequest,
  SwapPartsData,
  SwapPreviewAllRequest,
  SwapPreviewAllData,
  SwapJobStatus,
  SwapPartsIntensity,
} from './types';

/**
 * Face Swapを実行
 *
 * @param request - 現在の画像と理想の画像（Base64）
 * @returns ステータスとスワップ結果画像
 *
 * @example
 * ```typescript
 * const result = await generateSwap({
 *   current_image: base64Current,
 *   ideal_image: base64Ideal,
 * });
 * if (result.status === 'completed') {
 *   const imgSrc = `data:image/png;base64,${result.swapped_image}`;
 * }
 * ```
 */
export async function generateSwap(
  request: SwapGenerateRequest
): Promise<SwapResultData> {
  return apiPost<SwapResultData>('/api/v1/swap/generate', request, {
    timeout: 120000, // 120秒（Replicate APIの処理時間を考慮）
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
 * Face Swapを実行（同期的に完了まで待機）
 *
 * generateSwapのエイリアス。以前のポーリングAPIとの互換性のため提供。
 *
 * @param request - リクエストデータ
 * @param options - コールバックオプション（互換性のため）
 * @returns 完了時の結果データ（swapped_image含む）
 *
 * @example
 * ```typescript
 * const result = await swapAndWait({
 *   current_image: base64Current,
 *   ideal_image: base64Ideal,
 * });
 * setSwappedImage(result.swapped_image);
 * ```
 */
export async function swapAndWait(
  request: SwapGenerateRequest,
  options: {
    onProgress?: (status: SwapJobStatus) => void;
  } = {}
): Promise<SwapResultData> {
  // コールバックがあれば開始を通知
  if (options.onProgress) {
    options.onProgress('processing');
  }

  // 同期的にSwapを実行（結果が直接返される）
  const result = await generateSwap(request);

  // 完了を通知
  if (options.onProgress) {
    options.onProgress(result.status);
  }

  return result;
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
  eyes: 1.0,
  nose: 1.0,
  lips: 1.0,
};

/**
 * パーツ強度をリセット（全て0）
 */
export const ZERO_PARTS_INTENSITY: SwapPartsIntensity = {
  eyes: 0,
  nose: 0,
  lips: 0,
};
