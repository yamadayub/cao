/**
 * モーフィングAPI
 */

import { apiPostFormData } from './client';
import { MorphData, StagedMorphData, PartsBlendData, PartsSelection } from './types';

/**
 * デフォルトの段階値
 */
export const DEFAULT_STAGES = [0, 0.25, 0.5, 0.75, 1.0];

/**
 * 2つの顔画像をモーフィングする
 *
 * @param current - 現在の顔画像（JPEG/PNG、最大10MB）
 * @param ideal - 理想の顔画像（JPEG/PNG、最大10MB）
 * @param progress - 変化度合い（0.0 = 現在、1.0 = 理想、デフォルト: 0.5）
 * @returns モーフィング結果（Base64エンコードされた画像）
 * @throws ApiError - 顔未検出、画像フォーマットエラーなど
 *
 * @example
 * ```typescript
 * try {
 *   const result = await morphImages(currentFile, idealFile, 0.5);
 *   const imgSrc = `data:image/png;base64,${result.image}`;
 *   document.querySelector('img')!.src = imgSrc;
 * } catch (error) {
 *   console.error('Morphing failed:', error);
 * }
 * ```
 */
export async function morphImages(
  current: File,
  ideal: File,
  progress: number = 0.5
): Promise<MorphData> {
  const formData = new FormData();
  formData.append('current_image', current);
  formData.append('ideal_image', ideal);
  formData.append('progress', progress.toString());

  return apiPostFormData<MorphData>('/api/v1/morph', formData);
}

/**
 * 2つの顔画像を複数段階でモーフィングする
 *
 * @param current - 現在の顔画像（JPEG/PNG、最大10MB）
 * @param ideal - 理想の顔画像（JPEG/PNG、最大10MB）
 * @param stages - 生成する段階の配列（デフォルト: [0, 0.25, 0.5, 0.75, 1.0]）
 * @returns 段階別モーフィング結果
 * @throws ApiError - 顔未検出、画像フォーマットエラーなど
 *
 * @example
 * ```typescript
 * try {
 *   const result = await morphStages(currentFile, idealFile);
 *   result.images.forEach(({ progress, image }) => {
 *     console.log(`Progress ${progress * 100}%:`, image.substring(0, 50));
 *   });
 * } catch (error) {
 *   console.error('Staged morphing failed:', error);
 * }
 * ```
 */
export async function morphStages(
  current: File,
  ideal: File,
  stages: number[] = DEFAULT_STAGES
): Promise<StagedMorphData> {
  const formData = new FormData();
  formData.append('current_image', current);
  formData.append('ideal_image', ideal);
  formData.append('stages', JSON.stringify(stages));

  // 段階的モーフィングは時間がかかるため、長めのタイムアウトを設定
  return apiPostFormData<StagedMorphData>('/api/v1/morph/stages', formData, {
    timeout: 60000, // 60秒
  });
}

/**
 * Base64画像データをDataURLに変換するユーティリティ
 *
 * @param base64 - Base64エンコードされた画像データ
 * @param format - 画像フォーマット（デフォルト: 'png'）
 * @returns DataURL形式の文字列
 *
 * @example
 * ```typescript
 * const result = await morphImages(currentFile, idealFile);
 * const dataUrl = toDataUrl(result.image);
 * document.querySelector('img')!.src = dataUrl;
 * ```
 */
export function toDataUrl(base64: string, format: string = 'png'): string {
  // 既にdata:形式の場合はそのまま返す
  if (base64.startsWith('data:')) {
    return base64;
  }
  return `data:image/${format};base64,${base64}`;
}

/**
 * ブレンドメソッドの型
 */
export type BlendMethod = '2d' | '3d' | 'auto';

/**
 * 理想の顔から選択したパーツを現在の顔にブレンドする
 *
 * @param current - 現在の顔画像（JPEG/PNG、最大10MB）
 * @param ideal - 理想の顔画像（JPEG/PNG、最大10MB）
 * @param parts - ブレンドするパーツの選択
 * @param method - ブレンド方式（'2d', '3d', 'auto'）。デフォルトは'auto'
 *                 'auto'は3Dが利用可能な場合は3D、それ以外は2Dを使用
 * @returns ブレンド結果（Base64エンコードされた画像）
 * @throws ApiError - 顔未検出、画像フォーマットエラーなど
 *
 * @example
 * ```typescript
 * try {
 *   const result = await blendParts(currentFile, idealFile, {
 *     left_eye: true,
 *     right_eye: true,
 *     nose: true,
 *     lips: false,
 *     left_eyebrow: false,
 *     right_eyebrow: false,
 *   }, '3d');
 *   const imgSrc = `data:image/png;base64,${result.image}`;
 *   document.querySelector('img')!.src = imgSrc;
 * } catch (error) {
 *   console.error('Parts blend failed:', error);
 * }
 * ```
 */
export async function blendParts(
  current: File,
  ideal: File,
  parts: PartsSelection,
  method: BlendMethod = 'auto'
): Promise<PartsBlendData> {
  const formData = new FormData();
  formData.append('current_image', current);
  formData.append('ideal_image', ideal);
  formData.append('parts', JSON.stringify(parts));
  formData.append('method', method);

  // 3Dブレンドは特に時間がかかるため、より長めのタイムアウトを設定
  return apiPostFormData<PartsBlendData>('/api/v1/blend/parts', formData, {
    timeout: 120000, // 120秒（3Dは処理に時間がかかる）
  });
}
