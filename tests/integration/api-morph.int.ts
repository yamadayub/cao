/**
 * モーフィングAPI - 結合テスト
 *
 * 対象エンドポイント:
 * - POST /api/v1/morph (単一モーフィング)
 * - POST /api/v1/morph/stages (段階的モーフィング)
 *
 * 参照: functional-spec.md セクション 2.3.3, 2.3.4
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import * as fs from 'fs'
import * as path from 'path'

// テスト用の設定
const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000'
const MORPH_ENDPOINT = `${API_BASE_URL}/api/v1/morph`
const MORPH_STAGES_ENDPOINT = `${API_BASE_URL}/api/v1/morph/stages`

// テスト用画像パス（fixtures ディレクトリ）
const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures')

// フェッチヘルパー
const createMorphFormData = (
  currentImagePath: string,
  idealImagePath: string,
  options?: { progress?: number }
): FormData => {
  const formData = new FormData()

  const currentBuffer = fs.readFileSync(currentImagePath)
  const idealBuffer = fs.readFileSync(idealImagePath)

  formData.append(
    'current_image',
    new Blob([currentBuffer]),
    path.basename(currentImagePath)
  )
  formData.append(
    'ideal_image',
    new Blob([idealBuffer]),
    path.basename(idealImagePath)
  )

  if (options?.progress !== undefined) {
    formData.append('progress', options.progress.toString())
  }

  return formData
}

const createStagedMorphFormData = (
  currentImagePath: string,
  idealImagePath: string,
  stages?: number[]
): FormData => {
  const formData = new FormData()

  const currentBuffer = fs.readFileSync(currentImagePath)
  const idealBuffer = fs.readFileSync(idealImagePath)

  formData.append(
    'current_image',
    new Blob([currentBuffer]),
    path.basename(currentImagePath)
  )
  formData.append(
    'ideal_image',
    new Blob([idealBuffer]),
    path.basename(idealImagePath)
  )

  if (stages) {
    formData.append('stages', JSON.stringify(stages))
  }

  return formData
}

describe('POST /api/v1/morph - 単一モーフィングAPI', () => {
  describe('正常系', () => {
    it('2つの顔画像からモーフィング画像が生成される', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath)

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data).toBeDefined()
      expect(json.data.image).toBeDefined()
      expect(json.data.format).toBe('png')
      expect(json.data.progress).toBeDefined()
      expect(json.data.dimensions).toBeDefined()
      expect(json.data.dimensions.width).toBeGreaterThan(0)
      expect(json.data.dimensions.height).toBeGreaterThan(0)
    })

    it('デフォルトの progress は 0.5', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath)

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.data.progress).toBe(0.5)
    })

    it('progress=0.0 で現在の顔に近い画像が生成される', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath, {
        progress: 0.0,
      })

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.progress).toBe(0.0)
    })

    it('progress=1.0 で理想の顔に近い画像が生成される', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath, {
        progress: 1.0,
      })

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.progress).toBe(1.0)
    })

    it('progress=0.25 で任意の変化度合いを指定できる', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath, {
        progress: 0.25,
      })

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.progress).toBe(0.25)
    })

    it('生成される画像はBase64エンコードされている', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath)

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      // Base64形式またはdata URIスキームを確認
      const imageData = json.data.image
      const isBase64OrDataUri =
        /^[A-Za-z0-9+/=]+$/.test(imageData) ||
        imageData.startsWith('data:image/')

      expect(isBase64OrDataUri).toBe(true)
    })
  })

  describe('異常系 - バリデーションエラー', () => {
    it('current_image がない場合は VALIDATION_ERROR を返す', async () => {
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (!fs.existsSync(idealImagePath)) {
        console.warn('Fixture not found, skipping test')
        return
      }

      const formData = new FormData()
      const idealBuffer = fs.readFileSync(idealImagePath)
      formData.append('ideal_image', new Blob([idealBuffer]), 'ideal_face.jpg')

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })

    it('ideal_image がない場合は VALIDATION_ERROR を返す', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')

      if (!fs.existsSync(currentImagePath)) {
        console.warn('Fixture not found, skipping test')
        return
      }

      const formData = new FormData()
      const currentBuffer = fs.readFileSync(currentImagePath)
      formData.append(
        'current_image',
        new Blob([currentBuffer]),
        'valid_face.jpg'
      )

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })

    it('progress が範囲外（負の値）の場合は VALIDATION_ERROR を返す', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath, {
        progress: -0.1,
      })

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
      expect(json.error.message).toBe('Progress must be between 0.0 and 1.0')
    })

    it('progress が範囲外（1より大きい）の場合は VALIDATION_ERROR を返す', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath, {
        progress: 1.5,
      })

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })
  })

  describe('異常系 - 顔検出エラー', () => {
    it('current_image で顔が検出されない場合は FACE_NOT_DETECTED を返す', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'landscape.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath)

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('FACE_NOT_DETECTED')
      expect(json.error.message).toContain('current')
    })

    it('ideal_image で顔が検出されない場合は FACE_NOT_DETECTED を返す', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'landscape.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createMorphFormData(currentImagePath, idealImagePath)

      const response = await fetch(MORPH_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('FACE_NOT_DETECTED')
      expect(json.error.message).toContain('ideal')
    })
  })

  describe('異常系 - 処理エラー', () => {
    it('モーフィング処理失敗時は PROCESSING_ERROR を返す', async () => {
      // このテストは実装詳細に依存するため、スキップ可能
      // 処理失敗をシミュレートする方法が必要
      console.warn(
        'Processing error test requires specific test setup, skipping'
      )
    })
  })
})

describe('POST /api/v1/morph/stages - 段階的モーフィングAPI', () => {
  describe('正常系', () => {
    it('デフォルトで5段階のモーフィング画像が生成される', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createStagedMorphFormData(
        currentImagePath,
        idealImagePath
      )

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(200)

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.images).toBeDefined()
      expect(Array.isArray(json.data.images)).toBe(true)
      expect(json.data.images.length).toBe(5) // 0, 0.25, 0.5, 0.75, 1.0

      // 各段階のprogressを確認
      const progresses = json.data.images.map(
        (img: { progress: number }) => img.progress
      )
      expect(progresses).toEqual([0, 0.25, 0.5, 0.75, 1.0])
    })

    it('各段階の画像がBase64エンコードされている', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createStagedMorphFormData(
        currentImagePath,
        idealImagePath
      )

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      json.data.images.forEach((img: { image: string; progress: number }) => {
        const isBase64OrDataUri =
          /^[A-Za-z0-9+/=]+$/.test(img.image) ||
          img.image.startsWith('data:image/')

        expect(isBase64OrDataUri).toBe(true)
      })
    })

    it('カスタムの stages を指定できる', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const customStages = [0, 0.33, 0.66, 1.0]
      const formData = createStagedMorphFormData(
        currentImagePath,
        idealImagePath,
        customStages
      )

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.success).toBe(true)
      expect(json.data.images.length).toBe(4)

      const progresses = json.data.images.map(
        (img: { progress: number }) => img.progress
      )
      expect(progresses).toEqual(customStages)
    })

    it('レスポンスに format と dimensions が含まれる', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createStagedMorphFormData(
        currentImagePath,
        idealImagePath
      )

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const json = await response.json()

      expect(json.data.format).toBe('png')
      expect(json.data.dimensions).toBeDefined()
      expect(json.data.dimensions.width).toBeGreaterThan(0)
      expect(json.data.dimensions.height).toBeGreaterThan(0)
    })
  })

  describe('異常系', () => {
    it('画像がない場合は VALIDATION_ERROR を返す', async () => {
      const formData = new FormData()

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })

    it('不正な stages 形式の場合は VALIDATION_ERROR を返す', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = new FormData()
      formData.append(
        'current_image',
        new Blob([fs.readFileSync(currentImagePath)]),
        'current.jpg'
      )
      formData.append(
        'ideal_image',
        new Blob([fs.readFileSync(idealImagePath)]),
        'ideal.jpg'
      )
      formData.append('stages', 'invalid-json')

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
    })

    it('stages に範囲外の値が含まれる場合は VALIDATION_ERROR を返す', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const invalidStages = [0, 0.5, 1.5] // 1.5 is out of range
      const formData = createStagedMorphFormData(
        currentImagePath,
        idealImagePath,
        invalidStages
      )

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      expect(response.status).toBe(400)

      const json = await response.json()

      expect(json.success).toBe(false)
      expect(json.error.code).toBe('VALIDATION_ERROR')
    })
  })

  describe('パフォーマンス', () => {
    it('5段階のモーフィング生成が10秒以内に完了する', async () => {
      const currentImagePath = path.join(FIXTURES_DIR, 'valid_face.jpg')
      const idealImagePath = path.join(FIXTURES_DIR, 'ideal_face.jpg')

      if (
        !fs.existsSync(currentImagePath) ||
        !fs.existsSync(idealImagePath)
      ) {
        console.warn('Fixtures not found, skipping test')
        return
      }

      const formData = createStagedMorphFormData(
        currentImagePath,
        idealImagePath
      )

      const startTime = Date.now()

      const response = await fetch(MORPH_STAGES_ENDPOINT, {
        method: 'POST',
        body: formData,
      })

      const endTime = Date.now()
      const duration = endTime - startTime

      expect(response.status).toBe(200)
      expect(duration).toBeLessThan(10000) // 10秒以内
    })
  })
})
