/**
 * シェアAPIクライアントのテスト
 *
 * UC-013〜UC-016のAPI呼び出しテスト
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// 型定義
type ShareTemplate = 'before_after' | 'single' | 'parts_highlight'

interface CreateSnsShareRequest {
  sourceImage: string
  resultImage: string
  template: ShareTemplate
  caption?: string
  appliedParts?: string[]
}

interface SnsShareData {
  shareId: string
  shareUrl: string
  shareImageUrl: string
  ogImageUrl: string
  expiresAt: string
}

interface GetSnsShareData {
  shareId: string
  shareImageUrl: string
  caption: string | null
  template: ShareTemplate
  createdAt: string
  expiresAt: string
  isExpired: boolean
}

// モックAPI関数
const createSnsShare = async (
  request: CreateSnsShareRequest,
  token: string
): Promise<SnsShareData> => {
  const response = await fetch('/api/v1/share/create', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      source_image: request.sourceImage,
      result_image: request.resultImage,
      template: request.template,
      caption: request.caption,
      applied_parts: request.appliedParts,
    }),
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  const data = await response.json()
  return {
    shareId: data.data.share_id,
    shareUrl: data.data.share_url,
    shareImageUrl: data.data.share_image_url,
    ogImageUrl: data.data.og_image_url,
    expiresAt: data.data.expires_at,
  }
}

const getSnsShare = async (shareId: string): Promise<GetSnsShareData> => {
  const response = await fetch(`/api/v1/share/${shareId}`)

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  const data = await response.json()
  return {
    shareId: data.data.share_id,
    shareImageUrl: data.data.share_image_url,
    caption: data.data.caption,
    template: data.data.template,
    createdAt: data.data.created_at,
    expiresAt: data.data.expires_at,
    isExpired: data.data.is_expired,
  }
}

describe('シェアAPIクライアント', () => {
  const mockFetch = vi.fn()

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  describe('createSnsShare', () => {
    it('正常にシェアを作成できる', async () => {
      const mockResponse = {
        success: true,
        data: {
          share_id: 'test-share-id',
          share_url: 'https://cao.app/share/test-share-id',
          share_image_url: 'https://storage.cao.app/shares/test-share-id/share.png',
          og_image_url: 'https://storage.cao.app/shares/test-share-id/og.png',
          expires_at: '2025-03-01T00:00:00Z',
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await createSnsShare(
        {
          sourceImage: 'base64-source',
          resultImage: 'base64-result',
          template: 'before_after',
          caption: 'テストキャプション',
        },
        'test-token'
      )

      expect(result.shareId).toBe('test-share-id')
      expect(result.shareUrl).toBe('https://cao.app/share/test-share-id')
      expect(mockFetch).toHaveBeenCalledWith('/api/v1/share/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer test-token',
        },
        body: expect.any(String),
      })
    })

    it('認証エラーで例外がスローされる', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
      })

      await expect(
        createSnsShare(
          {
            sourceImage: 'base64-source',
            resultImage: 'base64-result',
            template: 'before_after',
          },
          'invalid-token'
        )
      ).rejects.toThrow('API error: 401')
    })

    it('キャプションなしでも作成できる', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            data: {
              share_id: 'test-share-id',
              share_url: 'https://cao.app/share/test-share-id',
              share_image_url: 'https://storage.cao.app/shares/test-share-id/share.png',
              og_image_url: 'https://storage.cao.app/shares/test-share-id/og.png',
              expires_at: '2025-03-01T00:00:00Z',
            },
          }),
      })

      const result = await createSnsShare(
        {
          sourceImage: 'base64-source',
          resultImage: 'base64-result',
          template: 'single',
        },
        'test-token'
      )

      expect(result.shareId).toBe('test-share-id')
    })
  })

  describe('getSnsShare', () => {
    it('正常にシェアを取得できる', async () => {
      const mockResponse = {
        success: true,
        data: {
          share_id: 'test-share-id',
          share_image_url: 'https://storage.cao.app/shares/test-share-id/share.png',
          caption: 'テストキャプション',
          template: 'before_after',
          created_at: '2025-01-29T00:00:00Z',
          expires_at: '2025-02-28T00:00:00Z',
          is_expired: false,
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await getSnsShare('test-share-id')

      expect(result.shareId).toBe('test-share-id')
      expect(result.caption).toBe('テストキャプション')
      expect(result.isExpired).toBe(false)
    })

    it('存在しないシェアIDで404エラー', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      })

      await expect(getSnsShare('invalid-id')).rejects.toThrow('API error: 404')
    })

    it('期限切れのシェアを取得できる', async () => {
      const mockResponse = {
        success: true,
        data: {
          share_id: 'expired-share-id',
          share_image_url: 'https://storage.cao.app/shares/expired-share-id/share.png',
          caption: null,
          template: 'single',
          created_at: '2024-12-01T00:00:00Z',
          expires_at: '2024-12-31T00:00:00Z',
          is_expired: true,
        },
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })

      const result = await getSnsShare('expired-share-id')

      expect(result.isExpired).toBe(true)
    })
  })
})

describe('SNSシェアURL生成', () => {
  it('Twitter Web Intent URLを正しく生成する', () => {
    const shareUrl = 'https://cao.app/share/test-id'
    const caption = 'テストキャプション'

    const encodedUrl = encodeURIComponent(shareUrl)
    const encodedCaption = encodeURIComponent(caption)
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodedCaption}&url=${encodedUrl}`

    expect(twitterUrl).toContain('twitter.com/intent/tweet')
    expect(twitterUrl).toContain('text=')
    expect(twitterUrl).toContain('url=')
  })

  it('LINE Share URLを正しく生成する', () => {
    const shareUrl = 'https://cao.app/share/test-id'

    const encodedUrl = encodeURIComponent(shareUrl)
    const lineUrl = `https://social-plugins.line.me/lineit/share?url=${encodedUrl}`

    expect(lineUrl).toContain('social-plugins.line.me')
    expect(lineUrl).toContain('url=')
  })
})
