/**
 * Face Swap APIのテスト
 *
 * - APIレスポンスの正しいパース
 * - エラーハンドリング
 * - 大きなレスポンスの処理
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// モックfetch
const mockFetch = vi.fn()
global.fetch = mockFetch

// テスト用の小さなbase64画像（1x1ピクセル透明PNG）
const TINY_PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

// モックAPI関数（実際のクライアントと同じロジック）
async function generateSwap(request: { current_image: string; ideal_image: string }) {
  const response = await fetch('http://localhost:8000/api/v1/swap/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.error?.message || 'API error')
  }

  const data = await response.json()

  if (!data.success) {
    throw new Error(data.error?.message || 'Unknown error')
  }

  return data.data
}

describe('Face Swap API', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('generateSwap', () => {
    it('正常なレスポンスを正しくパースできる', async () => {
      const mockResponse = {
        success: true,
        data: {
          status: 'completed',
          swapped_image: TINY_PNG_BASE64,
          error: null,
        },
        meta: null,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await generateSwap({
        current_image: TINY_PNG_BASE64,
        ideal_image: TINY_PNG_BASE64,
      })

      expect(result.status).toBe('completed')
      expect(result.swapped_image).toBe(TINY_PNG_BASE64)
      expect(result.error).toBeNull()
    })

    it('大きなbase64レスポンス（2MB相当）を処理できる', async () => {
      // 2MB相当のbase64文字列を生成（実際の画像ではない）
      const largeBase64 = 'A'.repeat(2 * 1024 * 1024)

      const mockResponse = {
        success: true,
        data: {
          status: 'completed',
          swapped_image: largeBase64,
          error: null,
        },
        meta: null,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await generateSwap({
        current_image: TINY_PNG_BASE64,
        ideal_image: TINY_PNG_BASE64,
      })

      expect(result.status).toBe('completed')
      expect(result.swapped_image).toBe(largeBase64)
      expect(result.swapped_image.length).toBeGreaterThanOrEqual(2 * 1024 * 1024)
    })

    it('APIエラー（400）を正しくハンドリングする', async () => {
      const mockErrorResponse = {
        success: false,
        error: {
          code: 'VALIDATION_ERROR',
          message: '画像形式が不正です',
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve(mockErrorResponse),
      })

      await expect(
        generateSwap({
          current_image: 'invalid',
          ideal_image: 'invalid',
        })
      ).rejects.toThrow('画像形式が不正です')
    })

    it('レート制限エラー（429）を正しくハンドリングする', async () => {
      const mockErrorResponse = {
        success: false,
        error: {
          code: 'RATE_LIMITED',
          message: 'サーバーが混雑しています。しばらく待ってからお試しください。',
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve(mockErrorResponse),
      })

      await expect(
        generateSwap({
          current_image: TINY_PNG_BASE64,
          ideal_image: TINY_PNG_BASE64,
        })
      ).rejects.toThrow('サーバーが混雑しています')
    })

    it('サーバーエラー（500）を正しくハンドリングする', async () => {
      const mockErrorResponse = {
        success: false,
        error: {
          code: 'PROCESSING_ERROR',
          message: 'Face swap処理に失敗しました',
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve(mockErrorResponse),
      })

      await expect(
        generateSwap({
          current_image: TINY_PNG_BASE64,
          ideal_image: TINY_PNG_BASE64,
        })
      ).rejects.toThrow('Face swap処理に失敗しました')
    })

    it('ネットワークエラーを正しくハンドリングする', async () => {
      mockFetch.mockRejectedValueOnce(new TypeError('Failed to fetch'))

      await expect(
        generateSwap({
          current_image: TINY_PNG_BASE64,
          ideal_image: TINY_PNG_BASE64,
        })
      ).rejects.toThrow('Failed to fetch')
    })

    it('JSONパースエラーを正しくハンドリングする', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.reject(new SyntaxError('Unexpected token')),
      })

      await expect(
        generateSwap({
          current_image: TINY_PNG_BASE64,
          ideal_image: TINY_PNG_BASE64,
        })
      ).rejects.toThrow()
    })

    it('swapped_imageがnullの場合はnullを返す', async () => {
      const mockResponse = {
        success: true,
        data: {
          status: 'failed',
          swapped_image: null,
          error: 'Face not detected',
        },
        meta: null,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await generateSwap({
        current_image: TINY_PNG_BASE64,
        ideal_image: TINY_PNG_BASE64,
      })

      expect(result.status).toBe('failed')
      expect(result.swapped_image).toBeNull()
      expect(result.error).toBe('Face not detected')
    })
  })
})
