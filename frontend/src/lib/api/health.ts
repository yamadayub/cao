/**
 * ヘルスチェックAPI
 */

import { apiGet } from './client';
import { HealthData } from './types';

/**
 * APIサーバーのヘルスチェックを実行
 *
 * @returns ヘルスチェック結果（ステータス、バージョン、タイムスタンプ）
 * @throws NetworkError - サーバーに接続できない場合
 *
 * @example
 * ```typescript
 * try {
 *   const health = await checkHealth();
 *   console.log('API Status:', health.status);
 *   console.log('Version:', health.version);
 * } catch (error) {
 *   console.error('API server is not responding');
 * }
 * ```
 */
export async function checkHealth(): Promise<HealthData> {
  return apiGet<HealthData>('/health');
}
