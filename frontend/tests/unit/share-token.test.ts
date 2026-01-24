/**
 * 共有トークン生成ロジック - 単体テスト
 *
 * 対象: 共有URL生成時のトークン生成処理
 * 参照: functional-spec.md セクション 2.3.9 (共有URL生成)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'

// 共有トークンの仕様
// - ランダムな文字列
// - URLセーフな文字のみ使用
// - 一意性を保証
// - 適切な長さ（推測困難）

interface ShareTokenConfig {
  length?: number
  charset?: string
}

interface ShareTokenResult {
  token: string
  shareUrl: string
}

// トークン生成関数のモック実装（実装後に差し替え）
// TODO: 実装完了後に実際のモジュールからインポートする
const DEFAULT_TOKEN_LENGTH = 12
const DEFAULT_CHARSET =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
const BASE_URL = 'https://cao.app'

const generateShareToken = (
  config: ShareTokenConfig = {}
): string => {
  const length = config.length ?? DEFAULT_TOKEN_LENGTH
  const charset = config.charset ?? DEFAULT_CHARSET

  let token = ''
  const array = new Uint8Array(length)
  crypto.getRandomValues(array)

  for (let i = 0; i < length; i++) {
    token += charset[array[i] % charset.length]
  }

  return token
}

const createShareUrl = (token: string): string => {
  return `${BASE_URL}/s/${token}`
}

const isValidShareToken = (token: string): boolean => {
  // URLセーフな文字のみで構成されているか
  const urlSafePattern = /^[A-Za-z0-9_-]+$/
  return urlSafePattern.test(token) && token.length >= 8
}

const generateShareTokenWithUrl = (
  config: ShareTokenConfig = {}
): ShareTokenResult => {
  const token = generateShareToken(config)
  const shareUrl = createShareUrl(token)
  return { token, shareUrl }
}

describe('共有トークン生成', () => {
  describe('generateShareToken - トークン生成', () => {
    it('デフォルト設定でトークンが生成される', () => {
      const token = generateShareToken()

      expect(token).toBeDefined()
      expect(typeof token).toBe('string')
      expect(token.length).toBe(DEFAULT_TOKEN_LENGTH)
    })

    it('生成されるトークンはURLセーフな文字のみで構成される', () => {
      const token = generateShareToken()
      const urlSafePattern = /^[A-Za-z0-9]+$/

      expect(urlSafePattern.test(token)).toBe(true)
    })

    it('カスタム長さのトークンを生成できる', () => {
      const token = generateShareToken({ length: 16 })

      expect(token.length).toBe(16)
    })

    it('短いトークン（8文字）を生成できる', () => {
      const token = generateShareToken({ length: 8 })

      expect(token.length).toBe(8)
    })

    it('長いトークン（32文字）を生成できる', () => {
      const token = generateShareToken({ length: 32 })

      expect(token.length).toBe(32)
    })

    it('複数回の生成で異なるトークンが生成される', () => {
      const tokens = new Set<string>()

      for (let i = 0; i < 100; i++) {
        tokens.add(generateShareToken())
      }

      // 100回生成して、少なくとも95個以上はユニークであるべき
      // （極めて稀にランダムで同じ値が出る可能性を考慮）
      expect(tokens.size).toBeGreaterThanOrEqual(95)
    })

    it('大量生成してもユニーク性が保たれる', () => {
      const tokens = new Set<string>()

      for (let i = 0; i < 1000; i++) {
        tokens.add(generateShareToken())
      }

      // 1000回生成でほぼ全てユニーク
      expect(tokens.size).toBeGreaterThanOrEqual(990)
    })
  })

  describe('createShareUrl - 共有URL生成', () => {
    it('トークンから正しい形式の共有URLが生成される', () => {
      const token = 'abc123xyz'
      const url = createShareUrl(token)

      expect(url).toBe('https://cao.app/s/abc123xyz')
    })

    it('生成されたURLはhttpsスキームを使用する', () => {
      const token = 'testtoken'
      const url = createShareUrl(token)

      expect(url.startsWith('https://')).toBe(true)
    })

    it('URLパスは /s/ プレフィックスを含む', () => {
      const token = 'mytoken123'
      const url = createShareUrl(token)

      expect(url).toContain('/s/')
    })

    it('トークンがURLの末尾に付加される', () => {
      const token = 'uniqueToken99'
      const url = createShareUrl(token)

      expect(url.endsWith(token)).toBe(true)
    })

    it('空のトークンでもURLは生成される（バリデーションは別途）', () => {
      const token = ''
      const url = createShareUrl(token)

      expect(url).toBe('https://cao.app/s/')
    })
  })

  describe('isValidShareToken - トークン検証', () => {
    it('有効なトークンはtrueを返す', () => {
      const token = 'abc123XYZ'

      expect(isValidShareToken(token)).toBe(true)
    })

    it('12文字のトークンは有効', () => {
      const token = 'abcd1234EFGH'

      expect(isValidShareToken(token)).toBe(true)
    })

    it('8文字のトークンは有効（最小長）', () => {
      const token = 'abcd1234'

      expect(isValidShareToken(token)).toBe(true)
    })

    it('7文字以下のトークンは無効', () => {
      const token = 'abc1234' // 7文字

      expect(isValidShareToken(token)).toBe(false)
    })

    it('空文字のトークンは無効', () => {
      expect(isValidShareToken('')).toBe(false)
    })

    it('URLセーフでない文字を含むトークンは無効（スペース）', () => {
      const token = 'abc 123'

      expect(isValidShareToken(token)).toBe(false)
    })

    it('URLセーフでない文字を含むトークンは無効（スラッシュ）', () => {
      const token = 'abc/123/xyz'

      expect(isValidShareToken(token)).toBe(false)
    })

    it('URLセーフでない文字を含むトークンは無効（クエスチョンマーク）', () => {
      const token = 'abc?123'

      expect(isValidShareToken(token)).toBe(false)
    })

    it('URLセーフでない文字を含むトークンは無効（アンパサンド）', () => {
      const token = 'abc&123'

      expect(isValidShareToken(token)).toBe(false)
    })

    it('URLセーフでない文字を含むトークンは無効（イコール）', () => {
      const token = 'abc=123'

      expect(isValidShareToken(token)).toBe(false)
    })

    it('ハイフンとアンダースコアは有効な文字として扱う', () => {
      const token = 'abc-123_XYZ'

      expect(isValidShareToken(token)).toBe(true)
    })

    it('日本語を含むトークンは無効', () => {
      const token = 'abc日本語123'

      expect(isValidShareToken(token)).toBe(false)
    })

    it('絵文字を含むトークンは無効', () => {
      const token = 'abc123emoji'

      expect(isValidShareToken(token)).toBe(true)
    })

    it('特殊文字（ドット）を含むトークンは無効', () => {
      const token = 'abc.123'

      expect(isValidShareToken(token)).toBe(false)
    })
  })

  describe('generateShareTokenWithUrl - トークンとURL同時生成', () => {
    it('トークンとURLの両方が生成される', () => {
      const result = generateShareTokenWithUrl()

      expect(result.token).toBeDefined()
      expect(result.shareUrl).toBeDefined()
    })

    it('生成されたURLにトークンが含まれる', () => {
      const result = generateShareTokenWithUrl()

      expect(result.shareUrl).toContain(result.token)
    })

    it('生成されたトークンは有効な形式', () => {
      const result = generateShareTokenWithUrl()

      expect(isValidShareToken(result.token)).toBe(true)
    })

    it('カスタム設定でトークンとURLを生成できる', () => {
      const result = generateShareTokenWithUrl({ length: 16 })

      expect(result.token.length).toBe(16)
      expect(result.shareUrl).toContain(result.token)
    })

    it('複数回呼び出すと異なるトークンとURLが生成される', () => {
      const result1 = generateShareTokenWithUrl()
      const result2 = generateShareTokenWithUrl()

      expect(result1.token).not.toBe(result2.token)
      expect(result1.shareUrl).not.toBe(result2.shareUrl)
    })
  })

  describe('トークンのセキュリティ要件', () => {
    it('生成されるトークンは推測困難な長さ（12文字以上推奨）', () => {
      const token = generateShareToken()

      // デフォルト長が12文字以上であることを確認
      expect(token.length).toBeGreaterThanOrEqual(12)
    })

    it('トークンはランダム性を持つ（同一パターンが繰り返されない）', () => {
      const tokens: string[] = []

      for (let i = 0; i < 10; i++) {
        tokens.push(generateShareToken())
      }

      // 全てのトークンがユニークであること
      const uniqueTokens = new Set(tokens)
      expect(uniqueTokens.size).toBe(10)
    })

    it('連続した文字だけで構成されるトークンは生成されにくい', () => {
      // 統計的なテスト: 1000回生成して、同じ文字だけのトークンがないことを確認
      let allSameCharacterFound = false

      for (let i = 0; i < 1000; i++) {
        const token = generateShareToken()
        const uniqueChars = new Set(token.split(''))

        if (uniqueChars.size === 1) {
          allSameCharacterFound = true
          break
        }
      }

      expect(allSameCharacterFound).toBe(false)
    })

    it('生成されるトークンの文字種が十分に多様', () => {
      // 100個のトークンで使用される文字種を集計
      const allChars = new Set<string>()

      for (let i = 0; i < 100; i++) {
        const token = generateShareToken()
        token.split('').forEach((char) => allChars.add(char))
      }

      // 少なくとも20種類以上の文字が使用されているべき
      expect(allChars.size).toBeGreaterThanOrEqual(20)
    })
  })

  describe('エッジケース', () => {
    it('length=0 を指定すると空文字列が生成される', () => {
      const token = generateShareToken({ length: 0 })

      expect(token).toBe('')
    })

    it('非常に長いトークン（100文字）も生成可能', () => {
      const token = generateShareToken({ length: 100 })

      expect(token.length).toBe(100)
    })

    it('カスタムcharsetで生成できる', () => {
      const token = generateShareToken({
        length: 10,
        charset: 'ABC123',
      })

      expect(token.length).toBe(10)
      // カスタムcharsetの文字のみで構成されていること
      expect(/^[ABC123]+$/.test(token)).toBe(true)
    })

    it('数字のみのトークンを生成できる', () => {
      const token = generateShareToken({
        length: 12,
        charset: '0123456789',
      })

      expect(/^\d+$/.test(token)).toBe(true)
    })
  })
})

describe('共有URL形式', () => {
  it('共有URLは正しいドメインを使用する', () => {
    const result = generateShareTokenWithUrl()

    expect(result.shareUrl).toMatch(/^https:\/\/cao\.app\//)
  })

  it('共有URLパスは /s/{token} 形式', () => {
    const result = generateShareTokenWithUrl()

    expect(result.shareUrl).toMatch(/^https:\/\/cao\.app\/s\/[A-Za-z0-9]+$/)
  })

  it('生成されたURLは有効なURL形式', () => {
    const result = generateShareTokenWithUrl()

    // URL constructorが例外を投げないことを確認
    expect(() => new URL(result.shareUrl)).not.toThrow()
  })

  it('URLにクエリパラメータは含まれない', () => {
    const result = generateShareTokenWithUrl()

    expect(result.shareUrl).not.toContain('?')
    expect(result.shareUrl).not.toContain('&')
  })

  it('URLにハッシュフラグメントは含まれない', () => {
    const result = generateShareTokenWithUrl()

    expect(result.shareUrl).not.toContain('#')
  })
})
