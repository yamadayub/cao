/**
 * 基本HTTPクライアント
 * 共通設定、エラーハンドリング、レスポンス型
 */

import {
  ApiResponse,
  ErrorCode,
  ErrorResponse,
  getErrorMessage,
} from './types';

// =============================================================================
// 設定
// =============================================================================

/**
 * API Base URL
 * 環境変数から取得、デフォルトはローカル開発用
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * デフォルトのリクエストタイムアウト (ミリ秒)
 */
const DEFAULT_TIMEOUT = 30000;

/**
 * リトライ設定
 */
const RETRY_CONFIG = {
  maxRetries: 3,
  baseDelay: 1000, // 1秒
  maxDelay: 4000,  // 4秒
};

// =============================================================================
// カスタムエラークラス
// =============================================================================

/**
 * APIエラークラス
 */
export class ApiError extends Error {
  public readonly code: ErrorCode;
  public readonly details?: Record<string, unknown>;
  public readonly statusCode?: number;

  constructor(
    code: ErrorCode,
    message: string,
    details?: Record<string, unknown>,
    statusCode?: number
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.details = details;
    this.statusCode = statusCode;
  }

  /**
   * ローカライズされたエラーメッセージを取得
   */
  get localizedMessage(): string {
    return getErrorMessage(this.code);
  }
}

/**
 * ネットワークエラークラス
 */
export class NetworkError extends Error {
  constructor(message: string = '通信エラーが発生しました。ネットワーク接続を確認してください') {
    super(message);
    this.name = 'NetworkError';
  }
}

/**
 * タイムアウトエラークラス
 */
export class TimeoutError extends Error {
  constructor(message: string = 'リクエストがタイムアウトしました') {
    super(message);
    this.name = 'TimeoutError';
  }
}

// =============================================================================
// ユーティリティ関数
// =============================================================================

/**
 * 指数バックオフで遅延を計算
 */
function calculateBackoffDelay(attempt: number): number {
  const delay = RETRY_CONFIG.baseDelay * Math.pow(2, attempt);
  return Math.min(delay, RETRY_CONFIG.maxDelay);
}

/**
 * 遅延を挿入
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * リトライ可能なエラーかどうかを判定
 */
function isRetryableError(error: unknown): boolean {
  if (error instanceof NetworkError) {
    return true;
  }
  if (error instanceof ApiError) {
    // 5xxエラーまたはレート制限はリトライ可能
    if (error.statusCode && error.statusCode >= 500) {
      return true;
    }
    if (error.code === 'RATE_LIMITED') {
      return true;
    }
  }
  return false;
}

/**
 * タイムアウト付きfetch
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new TimeoutError();
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// =============================================================================
// メインAPIクライアント
// =============================================================================

export interface ApiClientOptions extends RequestInit {
  timeout?: number;
  retries?: number;
  authToken?: string;
}

/**
 * APIクライアント - fetchのラッパー
 *
 * @param endpoint - APIエンドポイント（例: '/api/v1/analyze'）
 * @param options - リクエストオプション
 * @returns レスポンスデータ
 * @throws ApiError, NetworkError, TimeoutError
 */
export async function apiClient<T>(
  endpoint: string,
  options: ApiClientOptions = {}
): Promise<T> {
  const {
    timeout = DEFAULT_TIMEOUT,
    retries = RETRY_CONFIG.maxRetries,
    authToken,
    headers: customHeaders,
    ...fetchOptions
  } = options;

  const url = `${API_BASE_URL}${endpoint}`;

  // ヘッダーを構築
  const headers: HeadersInit = {
    ...customHeaders,
  };

  // Content-TypeがFormDataの場合は自動設定に任せる
  if (!(fetchOptions.body instanceof FormData)) {
    (headers as Record<string, string>)['Content-Type'] = 'application/json';
  }

  // 認証トークンがあれば追加
  if (authToken) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${authToken}`;
  }

  // リクエストIDを生成（トレーシング用）
  (headers as Record<string, string>)['X-Request-ID'] = crypto.randomUUID();

  const requestOptions: RequestInit = {
    ...fetchOptions,
    headers,
  };

  let lastError: Error | null = null;

  // リトライループ
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetchWithTimeout(url, requestOptions, timeout);

      // レスポンスをJSONとしてパース
      let data: ApiResponse<T>;
      try {
        data = await response.json();
      } catch {
        throw new ApiError(
          'INTERNAL_ERROR',
          'Invalid JSON response from server',
          undefined,
          response.status
        );
      }

      // エラーレスポンスの場合
      if (!response.ok || data.success === false) {
        const errorResponse = data as ErrorResponse;
        const errorDetail = errorResponse.error || {
          code: 'INTERNAL_ERROR' as ErrorCode,
          message: 'Unknown error',
        };

        throw new ApiError(
          errorDetail.code,
          errorDetail.message,
          errorDetail.details,
          response.status
        );
      }

      // 成功レスポンスのデータを返す
      return (data as { success: true; data: T }).data;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // ネットワークエラーを変換
      if (
        error instanceof TypeError &&
        error.message.includes('fetch')
      ) {
        lastError = new NetworkError();
      }

      // リトライ可能でない場合は即座にスロー
      if (!isRetryableError(lastError)) {
        throw lastError;
      }

      // 最後の試行の場合はスロー
      if (attempt === retries) {
        throw lastError;
      }

      // リトライ前に待機
      const delay = calculateBackoffDelay(attempt);
      await sleep(delay);
    }
  }

  // ここには到達しないはず
  throw lastError || new Error('Unknown error');
}

/**
 * GETリクエスト用ヘルパー
 */
export async function apiGet<T>(
  endpoint: string,
  options?: Omit<ApiClientOptions, 'method' | 'body'>
): Promise<T> {
  return apiClient<T>(endpoint, {
    ...options,
    method: 'GET',
  });
}

/**
 * POSTリクエスト用ヘルパー（JSON）
 */
export async function apiPost<T>(
  endpoint: string,
  body?: unknown,
  options?: Omit<ApiClientOptions, 'method' | 'body'>
): Promise<T> {
  return apiClient<T>(endpoint, {
    ...options,
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * POSTリクエスト用ヘルパー（FormData）
 */
export async function apiPostFormData<T>(
  endpoint: string,
  formData: FormData,
  options?: Omit<ApiClientOptions, 'method' | 'body' | 'headers'>
): Promise<T> {
  return apiClient<T>(endpoint, {
    ...options,
    method: 'POST',
    body: formData,
    // Content-Typeヘッダーは自動設定に任せる
  });
}

/**
 * DELETEリクエスト用ヘルパー
 */
export async function apiDelete<T>(
  endpoint: string,
  options?: Omit<ApiClientOptions, 'method' | 'body'>
): Promise<T> {
  return apiClient<T>(endpoint, {
    ...options,
    method: 'DELETE',
  });
}
