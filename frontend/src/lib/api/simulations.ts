/**
 * シミュレーションAPI
 * 認証が必要なエンドポイント
 */

import { apiGet, apiPost, apiDelete, ApiClientOptions } from './client';
import {
  CreateSimulationRequest,
  SimulationData,
  SimulationListData,
  DeleteData,
  ShareData,
  SharedSimulationData,
} from './types';

/**
 * 認証オプションを作成するヘルパー
 */
function withAuth(authToken: string): Pick<ApiClientOptions, 'authToken'> {
  return { authToken };
}

// =============================================================================
// シミュレーション保存・取得（要認証）
// =============================================================================

/**
 * シミュレーションを保存
 *
 * @param data - 保存するシミュレーションデータ
 * @param authToken - 認証トークン（Supabase JWT）
 * @returns 保存されたシミュレーション情報
 * @throws ApiError - UNAUTHORIZED（未認証）、VALIDATION_ERROR（バリデーションエラー）
 *
 * @example
 * ```typescript
 * const simulation = await createSimulation({
 *   current_image: currentBase64,
 *   ideal_image: idealBase64,
 *   result_images: morphResults.images.map(({ progress, image }) => ({
 *     progress,
 *     image,
 *   })),
 *   settings: { selected_progress: 0.5 },
 * }, authToken);
 * console.log('Saved simulation:', simulation.id);
 * ```
 */
export async function createSimulation(
  data: CreateSimulationRequest,
  authToken: string
): Promise<SimulationData> {
  return apiPost<SimulationData>('/api/v1/simulations', data, withAuth(authToken));
}

/**
 * シミュレーション一覧を取得
 *
 * @param authToken - 認証トークン
 * @param options - ページネーションオプション
 * @returns シミュレーション一覧とページネーション情報
 * @throws ApiError - UNAUTHORIZED（未認証）
 *
 * @example
 * ```typescript
 * const { simulations, pagination } = await getSimulations(authToken, {
 *   limit: 20,
 *   offset: 0,
 * });
 * simulations.forEach((sim) => {
 *   console.log(sim.id, sim.created_at);
 * });
 * ```
 */
export async function getSimulations(
  authToken: string,
  options: {
    limit?: number;
    offset?: number;
    sort?: string;
  } = {}
): Promise<SimulationListData> {
  const { limit = 20, offset = 0, sort = 'created_at:desc' } = options;

  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
    sort,
  });

  return apiGet<SimulationListData>(
    `/api/v1/simulations?${params.toString()}`,
    withAuth(authToken)
  );
}

/**
 * シミュレーション詳細を取得
 *
 * @param id - シミュレーションID（UUID）
 * @param authToken - 認証トークン
 * @returns シミュレーション詳細情報
 * @throws ApiError - UNAUTHORIZED（未認証）、NOT_FOUND（存在しない）
 *
 * @example
 * ```typescript
 * try {
 *   const simulation = await getSimulation(simulationId, authToken);
 *   console.log('Current image:', simulation.current_image_url);
 * } catch (error) {
 *   if (error.code === 'NOT_FOUND') {
 *     alert('シミュレーションが見つかりません');
 *   }
 * }
 * ```
 */
export async function getSimulation(
  id: string,
  authToken: string
): Promise<SimulationData> {
  return apiGet<SimulationData>(`/api/v1/simulations/${id}`, withAuth(authToken));
}

/**
 * シミュレーションを削除
 *
 * @param id - シミュレーションID（UUID）
 * @param authToken - 認証トークン
 * @returns 削除結果
 * @throws ApiError - UNAUTHORIZED（未認証）、NOT_FOUND（存在しない）
 *
 * @example
 * ```typescript
 * await deleteSimulation(simulationId, authToken);
 * console.log('Simulation deleted');
 * ```
 */
export async function deleteSimulation(
  id: string,
  authToken: string
): Promise<DeleteData> {
  return apiDelete<DeleteData>(`/api/v1/simulations/${id}`, withAuth(authToken));
}

// =============================================================================
// 共有機能
// =============================================================================

/**
 * シミュレーションの共有URLを生成
 *
 * @param id - シミュレーションID（UUID）
 * @param authToken - 認証トークン
 * @returns 共有トークンとURL
 * @throws ApiError - UNAUTHORIZED（未認証）、NOT_FOUND（存在しない）
 *
 * @example
 * ```typescript
 * const { share_url } = await createShareUrl(simulationId, authToken);
 * navigator.clipboard.writeText(share_url);
 * alert('URLをコピーしました');
 * ```
 */
export async function createShareUrl(
  id: string,
  authToken: string
): Promise<ShareData> {
  return apiPost<ShareData>(
    `/api/v1/simulations/${id}/share`,
    undefined,
    withAuth(authToken)
  );
}

/**
 * 共有されたシミュレーションを取得（認証不要）
 *
 * @param token - 共有トークン
 * @returns 共有シミュレーション情報（結果画像のみ）
 * @throws ApiError - NOT_FOUND（無効なトークン）
 *
 * @example
 * ```typescript
 * try {
 *   const shared = await getSharedSimulation(shareToken);
 *   shared.result_images.forEach(({ progress, url }) => {
 *     console.log(`Progress ${progress * 100}%:`, url);
 *   });
 * } catch (error) {
 *   if (error.code === 'NOT_FOUND') {
 *     // 無効または期限切れのURL
 *   }
 * }
 * ```
 */
export async function getSharedSimulation(
  token: string
): Promise<SharedSimulationData> {
  return apiGet<SharedSimulationData>(`/api/v1/shared/${token}`);
}
