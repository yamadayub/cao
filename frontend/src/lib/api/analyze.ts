/**
 * 顔分析API
 */

import { apiPostFormData } from './client';
import { AnalyzeData } from './types';

/**
 * 画像から顔を分析する
 *
 * @param image - 分析対象の画像ファイル（JPEG/PNG、最大10MB）
 * @returns 顔分析結果（顔領域、ランドマーク、画像情報）
 * @throws ApiError - 顔未検出、複数顔検出、画像フォーマットエラーなど
 *
 * @example
 * ```typescript
 * const fileInput = document.querySelector<HTMLInputElement>('#file-input');
 * const file = fileInput?.files?.[0];
 * if (file) {
 *   try {
 *     const result = await analyzeImage(file);
 *     if (result.face_detected) {
 *       console.log('Face found at:', result.face_region);
 *       console.log('Landmarks:', result.landmarks?.length);
 *     }
 *   } catch (error) {
 *     if (error instanceof ApiError && error.code === 'FACE_NOT_DETECTED') {
 *       alert('顔を検出できませんでした');
 *     }
 *   }
 * }
 * ```
 */
export async function analyzeImage(image: File): Promise<AnalyzeData> {
  const formData = new FormData();
  formData.append('image', image);

  return apiPostFormData<AnalyzeData>('/api/v1/analyze', formData);
}
