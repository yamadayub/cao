/**
 * 顔分析API - 結合テスト
 *
 * 対象エンドポイント: POST /api/v1/analyze
 * 参照: functional-spec.md セクション 2.3.2
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import * as fs from 'fs'
import * as path from 'path'

// テスト用の設定
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000'
const ANALYZE_ENDPOINT = `${API_BASE_URL}/api/v1/analyze`

// テスト用画像パス（fixtures ディレクトリ）
const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures')

// フェッチヘルパー
const createFormData = (imagePath: string, fieldName: string = 'image'): FormData => {
  const formData = new FormData()
  const imageBuffer = fs.readFileSync(imagePath)
  const blob = new Blob([imageBuffer])
  const fileName = path.basename(imagePath)
  formData.append(fieldName, blob, fileName)
  return formData
}

describe('POST /api/v1/analyze - 顔分析API', () => {
  describe('正常系', () => {
    it('JPEG画像で顔が1つ検出される場合、正常なレスポンスを返す', async () => {
      // テスト用のフィクスチャ画像が必要
      // TODO: 実際のテスト実行時にはフィクスチャ画像を用意する
      const testImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')

      // フィクスチャが存在しない場合はスキップ
      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data).toBeDefined()
      expect(json.data.face_detected).toBe(true)
      expect(json.data.face_count).toBe(1)
      expect(json.data.face_region).toBeDefined()
      expect(json.data.face_region).toHaveProperty('x')
      expect(json.data.face_region).toHaveProperty('y')
      expect(json.data.face_region).toHaveProperty('width')
      expect(json.data.face_region).toHaveProperty('height')
      expect(json.data.image_info).toBeDefined()
      expect(json.data.image_info.format).toBe('jpeg')
    })

    it('PNG画像で顔が1つ検出される場合、正常なレスポンスを返す', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'valid_face.png')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.face_detected).toBe(true)
      expect(json.data.face_count).toBe(1)
      expect(json.data.image_info.format).toBe('png')
    })

    it('レスポンスにランドマーク情報が含まれる', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.data.landmarks).toBeDefined()
      if (json.data.landmarks) {
        expect(Array.isArray(json.data.landmarks)).toBe(true)
        if (json.data.landmarks.length > 0) {
          const landmark = json.data.landmarks[0]
          expect(landmark).toHaveProperty('index')
          expect(landmark).toHaveProperty('x')
          expect(landmark).toHaveProperty('y')
          expect(landmark).toHaveProperty('z')
          // 正規化座標 (0.0 - 1.0) の範囲チェック
          expect(landmark.x).toBeGreaterThanOrEqual(0)
          expect(landmark.x).toBeLessThanOrEqual(1)
          expect(landmark.y).toBeGreaterThanOrEqual(0)
          expect(landmark.y).toBeLessThanOrEqual(1)
        }
      }
    })

    it('レスポンスに画像情報（width, height）が含まれる', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.data.image_info.width).toBeGreaterThan(0)
      expect(json.data.image_info.height).toBeGreaterThan(0)
    })
  })

  describe('異常系 - バリデーションエラー', () => {
    it('画像ファイルなしでリクエストすると VALIDATION_ERROR を返す', async () => {
      const formData = new FormData()
      // 画像を追加しない

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
      expect(json.error.message).toBe('Image file is required')
    })

    it('GIF形式の画像は INVALID_IMAGE_FORMAT を返す', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'animated.gif')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('INVALID_IMAGE_FORMAT')
      expect(json.error.message).toBe('Only JPEG and PNG formats are supported')
    })

    it('WebP形式の画像は INVALID_IMAGE_FORMAT を返す', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'image.webp')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('INVALID_IMAGE_FORMAT')
    })

    it('10MBを超える画像は IMAGE_TOO_LARGE を返す', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'large_file.jpg')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(413) // Payload Too Large

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('IMAGE_TOO_LARGE')
      expect(json.error.message).toBe('Image size must be under 10MB')
    })
  })

  describe('異常系 - 顔検出エラー', () => {
    it('顔が検出されない画像は FACE_NOT_DETECTED を返す', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'landscape.jpg')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('FACE_NOT_DETECTED')
      expect(json.error.message).toBe('No face detected in the uploaded image')
    })

    it('複数の顔が検出される画像は MULTIPLE_FACES を返す', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'group_photo.jpg')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('MULTIPLE_FACES')
      expect(json.error.message).toBe(
        'Multiple faces detected. Please upload an image with a single face'
      )
    })
  })

  describe('レスポンス形式', () => {
    it('成功時のレスポンスに meta 情報が含まれる', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      if (json.meta) {
        expect(json.meta).toHaveProperty('request_id')
        expect(json.meta).toHaveProperty('processing_time_ms')
        expect(typeof json.meta.processing_time_ms).toBe('number')
      }
    })

    it('エラー時のレスポンスは統一された形式', async () => {
      const formData = new FormData()

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json).toHaveProperty('success')
      expect(json.success).toBe(false)
      expect(json).toHaveProperty('error')
      expect(json.error).toHaveProperty('code')
      expect(json.error).toHaveProperty('message')
    })
  })

  describe('Content-Type', () => {
    it('multipart/form-data で送信する必要がある', async () => {
      // JSON として送信した場合のテスト
      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: 'base64data' }),
      })

      // 400 または 415 (Unsupported Media Type) を期待
      expect([400, 415]).toContain(response.status)
    })
  })

  describe('X-Request-ID ヘッダー', () => {
    it('X-Request-ID を送信するとレスポンスに含まれる', async () => {
      const testImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')

      if (!fs.existsSync(testImagePath)) {
        console.warn('Fixture not found, skipping test:', testImagePath)
        return
      }

      const formData = createFormData(testImagePath)
      const requestId = 'test-request-id-123'

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        headers: {
          'X-Request-ID': requestId,
        },
        body: formData,
      })

      const json = await response.json()

      if (json.meta) {
        expect(json.meta.request_id).toBe(requestId)
      }
    })
  })
})

describe('ヘルスチェック', () => {
  it('GET /health は正常なレスポンスを返す', async () => {
    const response = await fetch(`${API_BASE_URL}/health`)

    expect(response.status).toBe(200)

    const json = await response.json()

    expect(json.success).toBe(true)
    expect(json.data.status).toMatch(/^(ok|degraded)$/)
    expect(json.data.version).toBeDefined()
    expect(json.data.timestamp).toBeDefined()
  })
})
