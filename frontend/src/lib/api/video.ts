/**
 * Video Generation API
 *
 * モーフィング動画の生成
 */

import { apiPost } from './client';
import type { VideoGenerateData } from './types';

/**
 * モーフィング動画を生成
 *
 * @param sourceImage - Before画像（Base64エンコード）
 * @param resultImage - After画像（Base64エンコード）
 * @param authToken - 認証トークン
 * @returns 動画データ（URL, duration, format）
 * @throws ApiError - APIエラー
 */
export async function generateMorphVideo(
  sourceImage: string,
  resultImage: string,
  authToken: string
): Promise<VideoGenerateData> {
  return apiPost<VideoGenerateData>(
    '/api/v1/video/generate',
    {
      source_image: sourceImage,
      result_image: resultImage,
    },
    {
      authToken,
      timeout: 120000, // 2 minutes (video generation is heavy)
    }
  );
}

/**
 * ブレンド動画を生成（TikTok最適化・ギャップ最大化＋モーション）
 *
 * 2枚の画像（Before・After）からフラッシュトランジション＋ズームモーションの動画を生成
 *
 * @param currentImage - Before顔画像（Base64エンコード）
 * @param idealImage - 未使用（後方互換のため残存）
 * @param resultImage - After顔画像（Base64エンコード）
 * @param authToken - 認証トークン
 * @param transitionStyle - トランジション: "flash" (default), "blur", "snap"
 * @param motionStyle - モーション: "zoom" (default)
 * @returns 動画データ（URL, duration, format, quality_warning）
 * @throws ApiError - APIエラー
 */
export async function generateBlendVideo(
  currentImage: string,
  idealImage: string,
  resultImage: string,
  authToken: string,
  transitionStyle: string = 'flash',
  motionStyle: string = 'zoom'
): Promise<VideoGenerateData> {
  return apiPost<VideoGenerateData>(
    '/api/v1/video/blend',
    {
      current_image: currentImage,
      result_image: resultImage,
      transition_style: transitionStyle,
      motion_style: motionStyle,
    },
    {
      authToken,
      timeout: 120000,
    }
  );
}
