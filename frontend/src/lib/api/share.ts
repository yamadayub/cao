/**
 * SNS Share API
 *
 * UC-013〜UC-014: シェア画像の作成・取得
 */

import { apiGet, apiPost } from './client';
import type {
  SnsShareData,
  GetSnsShareData,
  ShareType,
} from './types';

// =============================================================================
// リクエスト型
// =============================================================================

export interface CreateSnsShareParams {
  sourceImage: string;        // Base64エンコード
  resultImage: string;        // Base64エンコード
  shareType: ShareType;       // 'before_after' | 'result_only'
}

// =============================================================================
// API関数
// =============================================================================

/**
 * SNSシェア画像を作成
 *
 * @param params - シェア作成パラメータ
 * @param authToken - 認証トークン
 * @returns シェアデータ
 * @throws ApiError - APIエラー
 */
export async function createSnsShare(
  params: CreateSnsShareParams,
  authToken: string
): Promise<SnsShareData> {
  return apiPost<SnsShareData>(
    '/api/v1/share/create',
    {
      source_image: params.sourceImage,
      result_image: params.resultImage,
      share_type: params.shareType,
    },
    { authToken }
  );
}

/**
 * SNSシェアデータを取得
 *
 * @param shareId - シェアID
 * @returns シェアデータ
 * @throws ApiError - APIエラー（404: NOT_FOUND）
 */
export async function getSnsShare(shareId: string): Promise<GetSnsShareData> {
  return apiGet<GetSnsShareData>(`/api/v1/share/${shareId}`);
}

// =============================================================================
// SNS共有URL生成ユーティリティ
// =============================================================================

/**
 * Twitter/X Web Intent URLを生成
 */
export function generateTwitterShareUrl(
  shareUrl: string,
  caption?: string
): string {
  const params = new URLSearchParams();
  if (caption) {
    params.set('text', caption);
  }
  params.set('url', shareUrl);
  return `https://twitter.com/intent/tweet?${params.toString()}`;
}

/**
 * LINE Share URLを生成
 */
export function generateLineShareUrl(shareUrl: string): string {
  const params = new URLSearchParams();
  params.set('url', shareUrl);
  return `https://social-plugins.line.me/lineit/share?${params.toString()}`;
}

/**
 * Instagram用のシェアデータを取得
 * (InstagramはWeb Intentをサポートしていないため、
 *  ネイティブアプリでの共有または画像ダウンロードを促す)
 */
export function getInstagramShareInfo(shareImageUrl: string) {
  return {
    imageUrl: shareImageUrl,
    message: '画像を保存してInstagramで共有してください',
  };
}
