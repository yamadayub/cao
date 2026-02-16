/**
 * 共有トークン生成ユーティリティ
 *
 * 共有URL生成時のトークン生成・検証処理を提供します。
 * 参照: functional-spec.md セクション 2.3.9 (共有URL生成)
 */

// ============================================
// 型定義
// ============================================

/**
 * トークン生成設定
 */
export interface ShareTokenConfig {
  /** トークンの長さ（デフォルト: 12） */
  length?: number
  /** 使用する文字セット（デフォルト: 英数字） */
  charset?: string
}

/**
 * トークン生成結果
 */
export interface ShareTokenResult {
  /** 生成されたトークン */
  token: string
  /** 共有URL */
  shareUrl: string
}

// ============================================
// 定数
// ============================================

/** デフォルトのトークン長 */
export const DEFAULT_TOKEN_LENGTH = 12

/** デフォルトの文字セット（URLセーフな英数字） */
export const DEFAULT_CHARSET =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

/** 共有URLのベースURL */
export const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || 'https://cao.style-elements.jp'

// ============================================
// 関数
// ============================================

/**
 * 共有トークンを生成する
 *
 * @param config - トークン生成設定
 * @returns 生成されたトークン文字列
 *
 * @example
 * ```ts
 * const token = generateShareToken()
 * // => 'abc123XYZ789'
 *
 * const customToken = generateShareToken({ length: 16 })
 * // => 'abcd1234EFGH5678'
 * ```
 */
export const generateShareToken = (config: ShareTokenConfig = {}): string => {
  const length = config.length ?? DEFAULT_TOKEN_LENGTH
  const charset = config.charset ?? DEFAULT_CHARSET

  if (length === 0) {
    return ''
  }

  let token = ''
  const array = new Uint8Array(length)
  crypto.getRandomValues(array)

  for (let i = 0; i < length; i++) {
    token += charset[array[i] % charset.length]
  }

  return token
}

/**
 * トークンから共有URLを生成する
 *
 * @param token - 共有トークン
 * @returns 共有URL
 *
 * @example
 * ```ts
 * const url = createShareUrl('abc123xyz')
 * // => 'https://cao.app/s/abc123xyz'
 * ```
 */
export const createShareUrl = (token: string): string => {
  return `${BASE_URL}/s/${token}`
}

/**
 * 共有トークンの形式を検証する
 *
 * 検証条件:
 * - URLセーフな文字のみ（英数字、ハイフン、アンダースコア）
 * - 8文字以上
 *
 * @param token - 検証するトークン
 * @returns 有効な場合はtrue、無効な場合はfalse
 *
 * @example
 * ```ts
 * isValidShareToken('abc123XYZ')  // => true
 * isValidShareToken('abc')        // => false (短すぎる)
 * isValidShareToken('abc/123')    // => false (無効な文字)
 * ```
 */
export const isValidShareToken = (token: string): boolean => {
  // URLセーフな文字のみで構成されているか（英数字、ハイフン、アンダースコア）
  const urlSafePattern = /^[A-Za-z0-9_-]+$/
  return urlSafePattern.test(token) && token.length >= 8
}

/**
 * 共有トークンとURLを同時に生成する
 *
 * @param config - トークン生成設定
 * @returns トークンと共有URLを含むオブジェクト
 *
 * @example
 * ```ts
 * const result = generateShareTokenWithUrl()
 * // => { token: 'abc123XYZ789', shareUrl: 'https://cao.app/s/abc123XYZ789' }
 * ```
 */
export const generateShareTokenWithUrl = (
  config: ShareTokenConfig = {}
): ShareTokenResult => {
  const token = generateShareToken(config)
  const shareUrl = createShareUrl(token)
  return { token, shareUrl }
}
