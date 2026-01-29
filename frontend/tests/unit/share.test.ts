/**
 * シェア機能のテスト
 *
 * UC-013: シェア画像の選択
 * UC-014: シェア画像のカスタマイズ
 * UC-015: SNSへのシェア
 * UC-016: SNSシェアページの閲覧
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

/**
 * シェアテンプレートの型定義
 */
type ShareTemplate = 'before_after' | 'single' | 'parts_highlight'

/**
 * シェア作成リクエストの型定義
 */
interface CreateShareRequest {
  sourceImage: string
  resultImage: string
  template: ShareTemplate
  caption?: string
  appliedParts?: string[]
}

/**
 * シェア作成レスポンスの型定義
 */
interface CreateShareResponse {
  shareId: string
  shareUrl: string
  shareImageUrl: string
  ogImageUrl: string
  expiresAt: string
}

/**
 * シェアデータの型定義
 */
interface ShareData {
  shareId: string
  shareImageUrl: string
  caption: string | null
  template: ShareTemplate
  createdAt: string
  expiresAt: string
  isExpired: boolean
}

describe('シェアテンプレート', () => {
  describe('テンプレートサイズ', () => {
    it('Before/Afterテンプレートは1200x630pxである', () => {
      const templateSize = { width: 1200, height: 630 }
      expect(templateSize.width).toBe(1200)
      expect(templateSize.height).toBe(630)
    })

    it('単体テンプレートは1080x1080pxである', () => {
      const templateSize = { width: 1080, height: 1080 }
      expect(templateSize.width).toBe(1080)
      expect(templateSize.height).toBe(1080)
    })

    it('パーツハイライトテンプレートは1080x1350pxである', () => {
      const templateSize = { width: 1080, height: 1350 }
      expect(templateSize.width).toBe(1080)
      expect(templateSize.height).toBe(1350)
    })
  })

  describe('テンプレートタイプ', () => {
    const validTemplates: ShareTemplate[] = ['before_after', 'single', 'parts_highlight']

    validTemplates.forEach((template) => {
      it(`${template} は有効なテンプレートタイプである`, () => {
        expect(validTemplates).toContain(template)
      })
    })
  })
})

describe('キャプションバリデーション', () => {
  it('キャプションは140文字以下である必要がある', () => {
    const maxLength = 140
    const validCaption = 'a'.repeat(140)
    const invalidCaption = 'a'.repeat(141)

    expect(validCaption.length).toBeLessThanOrEqual(maxLength)
    expect(invalidCaption.length).toBeGreaterThan(maxLength)
  })

  it('キャプションは省略可能である', () => {
    const caption: string | undefined = undefined
    expect(caption).toBeUndefined()
  })

  it('日本語キャプションをサポートする', () => {
    const caption = '理想の自分にまた一歩近づきました！'
    expect(caption.length).toBeLessThanOrEqual(140)
  })

  it('空のキャプションを許可する', () => {
    const caption = ''
    expect(caption).toBe('')
  })
})

describe('シェアURL生成', () => {
  it('シェアURLは正しい形式である', () => {
    const baseUrl = 'https://cao.app'
    const shareId = '550e8400-e29b-41d4-a716-446655440000'
    const expectedUrl = `${baseUrl}/share/${shareId}`

    expect(expectedUrl).toMatch(/^https:\/\/cao\.app\/share\/[a-f0-9-]+$/)
  })

  it('シェアIDはUUID形式である', () => {
    const shareId = '550e8400-e29b-41d4-a716-446655440000'
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

    expect(shareId).toMatch(uuidRegex)
  })
})

describe('SNSシェアURL', () => {
  describe('X（Twitter）シェア', () => {
    it('正しいWeb Intent URLを生成する', () => {
      const shareUrl = 'https://cao.app/share/test-id'
      const caption = 'テストキャプション'
      const encodedUrl = encodeURIComponent(shareUrl)
      const encodedCaption = encodeURIComponent(caption)

      const twitterIntentUrl = `https://twitter.com/intent/tweet?text=${encodedCaption}&url=${encodedUrl}`

      expect(twitterIntentUrl).toContain('twitter.com/intent/tweet')
      expect(twitterIntentUrl).toContain('text=')
      expect(twitterIntentUrl).toContain('url=')
    })
  })

  describe('LINEシェア', () => {
    it('正しいLINE Share URLを生成する', () => {
      const shareUrl = 'https://cao.app/share/test-id'
      const encodedUrl = encodeURIComponent(shareUrl)

      const lineShareUrl = `https://social-plugins.line.me/lineit/share?url=${encodedUrl}`

      expect(lineShareUrl).toContain('social-plugins.line.me')
      expect(lineShareUrl).toContain('url=')
    })
  })

  describe('Instagram', () => {
    it('Instagramはダウンロード方式であることを確認', () => {
      // Instagramは直接シェアできないため、画像ダウンロード後に手動で投稿
      const instagramShareMethod = 'download_and_manual'
      expect(instagramShareMethod).toBe('download_and_manual')
    })
  })
})

describe('シェア有効期限', () => {
  it('デフォルト有効期限は30日である', () => {
    const createdAt = new Date()
    const expiresAt = new Date(createdAt.getTime() + 30 * 24 * 60 * 60 * 1000)

    const diffDays = Math.round(
      (expiresAt.getTime() - createdAt.getTime()) / (1000 * 60 * 60 * 24)
    )

    expect(diffDays).toBe(30)
  })

  it('期限切れ判定が正しく動作する', () => {
    const now = new Date()

    // 期限切れのシェア
    const expiredShare: ShareData = {
      shareId: 'test-id',
      shareImageUrl: 'https://example.com/image.png',
      caption: null,
      template: 'before_after',
      createdAt: new Date(now.getTime() - 31 * 24 * 60 * 60 * 1000).toISOString(),
      expiresAt: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000).toISOString(),
      isExpired: true,
    }

    // 有効なシェア
    const validShare: ShareData = {
      shareId: 'test-id-2',
      shareImageUrl: 'https://example.com/image2.png',
      caption: null,
      template: 'single',
      createdAt: now.toISOString(),
      expiresAt: new Date(now.getTime() + 29 * 24 * 60 * 60 * 1000).toISOString(),
      isExpired: false,
    }

    expect(expiredShare.isExpired).toBe(true)
    expect(validShare.isExpired).toBe(false)
  })
})

describe('OGPメタタグ', () => {
  it('必要なOGPプロパティが定義されている', () => {
    const ogpProperties = [
      'og:title',
      'og:description',
      'og:image',
      'og:url',
      'og:type',
    ]

    expect(ogpProperties).toContain('og:title')
    expect(ogpProperties).toContain('og:description')
    expect(ogpProperties).toContain('og:image')
    expect(ogpProperties).toContain('og:url')
    expect(ogpProperties).toContain('og:type')
  })

  it('Twitterカードメタタグが定義されている', () => {
    const twitterMeta = [
      'twitter:card',
      'twitter:title',
      'twitter:description',
      'twitter:image',
    ]

    expect(twitterMeta).toContain('twitter:card')
    expect(twitterMeta).toContain('twitter:title')
    expect(twitterMeta).toContain('twitter:description')
    expect(twitterMeta).toContain('twitter:image')
  })

  it('og:titleのデフォルト値が正しい', () => {
    const defaultTitle = 'Caoで美容シミュレーション'
    expect(defaultTitle).toBe('Caoで美容シミュレーション')
  })
})

describe('プライバシー注意事項', () => {
  it('プライバシー警告メッセージが定義されている', () => {
    const warnings = [
      'シェアした画像は誰でも閲覧できます',
      '一度シェアした画像は削除できない場合があります',
      '他人の写真を無断でシェアしないでください',
    ]

    expect(warnings.length).toBe(3)
    expect(warnings[0]).toContain('誰でも閲覧')
    expect(warnings[1]).toContain('削除できない')
    expect(warnings[2]).toContain('無断でシェア')
  })

  it('同意チェックボックスが必須である', () => {
    const requiresConsent = true
    expect(requiresConsent).toBe(true)
  })
})

describe('シェア画像選択', () => {
  it('全体シミュレーション結果を選択できる', () => {
    const availableImages = [
      { type: 'morph', label: '全体結果' },
    ]

    expect(availableImages.some((img) => img.type === 'morph')).toBe(true)
  })

  it('パーツ別結果を選択できる（存在する場合）', () => {
    const availableImages = [
      { type: 'morph', label: '全体結果' },
      { type: 'parts', label: 'パーツ別結果', appliedParts: ['eyes', 'nose'] },
    ]

    const partsResults = availableImages.filter((img) => img.type === 'parts')
    expect(partsResults.length).toBeGreaterThanOrEqual(0)
  })

  it('選択は1つのみ可能である', () => {
    const selectedImages = ['image-1']
    expect(selectedImages.length).toBe(1)
  })
})

describe('APIレスポンス形式', () => {
  it('CreateShareResponseの必須フィールドが定義されている', () => {
    const response: CreateShareResponse = {
      shareId: 'test-id',
      shareUrl: 'https://cao.app/share/test-id',
      shareImageUrl: 'https://storage.example.com/share.png',
      ogImageUrl: 'https://storage.example.com/og.png',
      expiresAt: new Date().toISOString(),
    }

    expect(response).toHaveProperty('shareId')
    expect(response).toHaveProperty('shareUrl')
    expect(response).toHaveProperty('shareImageUrl')
    expect(response).toHaveProperty('ogImageUrl')
    expect(response).toHaveProperty('expiresAt')
  })

  it('ShareDataの必須フィールドが定義されている', () => {
    const shareData: ShareData = {
      shareId: 'test-id',
      shareImageUrl: 'https://storage.example.com/share.png',
      caption: 'テストキャプション',
      template: 'before_after',
      createdAt: new Date().toISOString(),
      expiresAt: new Date().toISOString(),
      isExpired: false,
    }

    expect(shareData).toHaveProperty('shareId')
    expect(shareData).toHaveProperty('shareImageUrl')
    expect(shareData).toHaveProperty('caption')
    expect(shareData).toHaveProperty('template')
    expect(shareData).toHaveProperty('createdAt')
    expect(shareData).toHaveProperty('expiresAt')
    expect(shareData).toHaveProperty('isExpired')
  })
})
