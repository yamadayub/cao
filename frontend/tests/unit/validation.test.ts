/**
 * 画像バリデーションロジック - 単体テスト
 *
 * 対象: 画像アップロード時のバリデーション処理
 * 参照: functional-spec.md セクション 3.3 (SCR-002)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'

// モック対象の型定義
interface ImageValidationResult {
  valid: boolean
  error?: {
    code: string
    message: string
  }
}

interface ImageInfo {
  width: number
  height: number
  format: 'jpeg' | 'png'
}

interface FaceDetectionResult {
  faceDetected: boolean
  faceCount: number
  faceRegion: {
    x: number
    y: number
    width: number
    height: number
  } | null
}

// バリデーション関数のモック実装（実装後に差し替え）
// TODO: 実装完了後に実際のモジュールからインポートする
const validateImageFormat = (file: File): ImageValidationResult => {
  const validTypes = ['image/jpeg', 'image/png']
  if (!validTypes.includes(file.type)) {
    return {
      valid: false,
      error: {
        code: 'INVALID_IMAGE_FORMAT',
        message: 'JPEG、PNG形式の画像をアップロードしてください',
      },
    }
  }
  return { valid: true }
}

const validateImageSize = (file: File, maxSizeMB: number = 10): ImageValidationResult => {
  const maxSizeBytes = maxSizeMB * 1024 * 1024
  if (file.size > maxSizeBytes) {
    return {
      valid: false,
      error: {
        code: 'IMAGE_TOO_LARGE',
        message: `画像サイズは${maxSizeMB}MB以下にしてください`,
      },
    }
  }
  return { valid: true }
}

const validateFaceDetection = (result: FaceDetectionResult): ImageValidationResult => {
  if (!result.faceDetected || result.faceCount === 0) {
    return {
      valid: false,
      error: {
        code: 'FACE_NOT_DETECTED',
        message: '顔を検出できませんでした。正面を向いた明るい写真をお使いください',
      },
    }
  }
  if (result.faceCount > 1) {
    return {
      valid: false,
      error: {
        code: 'MULTIPLE_FACES',
        message: '複数の顔が検出されました。1人のみ写った写真をお使いください',
      },
    }
  }
  return { valid: true }
}

// ファイルオブジェクトのモック作成ヘルパー
const createMockFile = (
  name: string,
  type: string,
  sizeInBytes: number
): File => {
  const blob = new Blob([''], { type })
  Object.defineProperty(blob, 'size', { value: sizeInBytes })
  Object.defineProperty(blob, 'name', { value: name })
  return blob as File
}

describe('画像バリデーション', () => {
  describe('validateImageFormat - 画像形式検証', () => {
    it('JPEG形式の画像は有効と判定される', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(true)
      expect(result.error).toBeUndefined()
    })

    it('PNG形式の画像は有効と判定される', () => {
      const file = createMockFile('test.png', 'image/png', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(true)
      expect(result.error).toBeUndefined()
    })

    it('GIF形式の画像は無効と判定される', () => {
      const file = createMockFile('test.gif', 'image/gif', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('INVALID_IMAGE_FORMAT')
      expect(result.error?.message).toBe(
        'JPEG、PNG形式の画像をアップロードしてください'
      )
    })

    it('WebP形式の画像は無効と判定される', () => {
      const file = createMockFile('test.webp', 'image/webp', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('INVALID_IMAGE_FORMAT')
    })

    it('BMP形式の画像は無効と判定される', () => {
      const file = createMockFile('test.bmp', 'image/bmp', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('INVALID_IMAGE_FORMAT')
    })

    it('SVG形式の画像は無効と判定される', () => {
      const file = createMockFile('test.svg', 'image/svg+xml', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('INVALID_IMAGE_FORMAT')
    })

    it('画像以外のファイル（PDF）は無効と判定される', () => {
      const file = createMockFile('test.pdf', 'application/pdf', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('INVALID_IMAGE_FORMAT')
    })

    it('テキストファイルは無効と判定される', () => {
      const file = createMockFile('test.txt', 'text/plain', 1024)
      const result = validateImageFormat(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('INVALID_IMAGE_FORMAT')
    })
  })

  describe('validateImageSize - ファイルサイズ検証', () => {
    it('10MB未満の画像は有効と判定される', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 5 * 1024 * 1024) // 5MB
      const result = validateImageSize(file)

      expect(result.valid).toBe(true)
      expect(result.error).toBeUndefined()
    })

    it('ちょうど10MBの画像は有効と判定される', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 10 * 1024 * 1024)
      const result = validateImageSize(file)

      expect(result.valid).toBe(true)
    })

    it('10MBを超える画像は無効と判定される', () => {
      const file = createMockFile(
        'test.jpg',
        'image/jpeg',
        10 * 1024 * 1024 + 1
      ) // 10MB + 1byte
      const result = validateImageSize(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('IMAGE_TOO_LARGE')
      expect(result.error?.message).toBe('画像サイズは10MB以下にしてください')
    })

    it('15MBの画像は無効と判定される', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 15 * 1024 * 1024)
      const result = validateImageSize(file)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('IMAGE_TOO_LARGE')
    })

    it('カスタム上限サイズを指定できる', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 6 * 1024 * 1024) // 6MB
      const result = validateImageSize(file, 5) // 5MB上限

      expect(result.valid).toBe(false)
      expect(result.error?.message).toBe('画像サイズは5MB以下にしてください')
    })

    it('0バイトのファイルは有効と判定される（空ファイルチェックは別途）', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 0)
      const result = validateImageSize(file)

      expect(result.valid).toBe(true)
    })

    it('1バイトの画像は有効と判定される', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 1)
      const result = validateImageSize(file)

      expect(result.valid).toBe(true)
    })
  })

  describe('validateFaceDetection - 顔検出結果検証', () => {
    it('1つの顔が検出された場合は有効と判定される', () => {
      const detectionResult: FaceDetectionResult = {
        faceDetected: true,
        faceCount: 1,
        faceRegion: { x: 100, y: 100, width: 200, height: 200 },
      }
      const result = validateFaceDetection(detectionResult)

      expect(result.valid).toBe(true)
      expect(result.error).toBeUndefined()
    })

    it('顔が検出されない場合は無効と判定される', () => {
      const detectionResult: FaceDetectionResult = {
        faceDetected: false,
        faceCount: 0,
        faceRegion: null,
      }
      const result = validateFaceDetection(detectionResult)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('FACE_NOT_DETECTED')
      expect(result.error?.message).toBe(
        '顔を検出できませんでした。正面を向いた明るい写真をお使いください'
      )
    })

    it('複数の顔が検出された場合は無効と判定される', () => {
      const detectionResult: FaceDetectionResult = {
        faceDetected: true,
        faceCount: 2,
        faceRegion: { x: 100, y: 100, width: 200, height: 200 },
      }
      const result = validateFaceDetection(detectionResult)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('MULTIPLE_FACES')
      expect(result.error?.message).toBe(
        '複数の顔が検出されました。1人のみ写った写真をお使いください'
      )
    })

    it('3人以上の顔が検出された場合も無効と判定される', () => {
      const detectionResult: FaceDetectionResult = {
        faceDetected: true,
        faceCount: 5,
        faceRegion: { x: 100, y: 100, width: 200, height: 200 },
      }
      const result = validateFaceDetection(detectionResult)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('MULTIPLE_FACES')
    })

    it('faceDetected=false かつ faceCount=1 の不整合ケースは無効と判定される', () => {
      // エッジケース: 実装によってはこのような状態が発生する可能性
      const detectionResult: FaceDetectionResult = {
        faceDetected: false,
        faceCount: 1,
        faceRegion: null,
      }
      const result = validateFaceDetection(detectionResult)

      expect(result.valid).toBe(false)
      expect(result.error?.code).toBe('FACE_NOT_DETECTED')
    })
  })

  describe('統合バリデーションシナリオ', () => {
    it('有効なJPEG画像（5MB、単一顔）は全バリデーションをパスする', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 5 * 1024 * 1024)
      const faceResult: FaceDetectionResult = {
        faceDetected: true,
        faceCount: 1,
        faceRegion: { x: 100, y: 100, width: 200, height: 200 },
      }

      const formatResult = validateImageFormat(file)
      const sizeResult = validateImageSize(file)
      const faceDetectionResult = validateFaceDetection(faceResult)

      expect(formatResult.valid).toBe(true)
      expect(sizeResult.valid).toBe(true)
      expect(faceDetectionResult.valid).toBe(true)
    })

    it('有効なPNG画像（3MB、単一顔）は全バリデーションをパスする', () => {
      const file = createMockFile('test.png', 'image/png', 3 * 1024 * 1024)
      const faceResult: FaceDetectionResult = {
        faceDetected: true,
        faceCount: 1,
        faceRegion: { x: 50, y: 50, width: 150, height: 150 },
      }

      const formatResult = validateImageFormat(file)
      const sizeResult = validateImageSize(file)
      const faceDetectionResult = validateFaceDetection(faceResult)

      expect(formatResult.valid).toBe(true)
      expect(sizeResult.valid).toBe(true)
      expect(faceDetectionResult.valid).toBe(true)
    })

    it('GIF画像は形式チェックで失敗する（サイズや顔検出は実行されない想定）', () => {
      const file = createMockFile('test.gif', 'image/gif', 1 * 1024 * 1024)

      const formatResult = validateImageFormat(file)

      expect(formatResult.valid).toBe(false)
      expect(formatResult.error?.code).toBe('INVALID_IMAGE_FORMAT')
    })

    it('巨大なJPEG画像はサイズチェックで失敗する', () => {
      const file = createMockFile('test.jpg', 'image/jpeg', 20 * 1024 * 1024) // 20MB

      const formatResult = validateImageFormat(file)
      const sizeResult = validateImageSize(file)

      expect(formatResult.valid).toBe(true) // 形式は有効
      expect(sizeResult.valid).toBe(false) // サイズは無効
      expect(sizeResult.error?.code).toBe('IMAGE_TOO_LARGE')
    })

    it('顔がない有効画像は顔検出チェックで失敗する', () => {
      const file = createMockFile('landscape.jpg', 'image/jpeg', 2 * 1024 * 1024)
      const faceResult: FaceDetectionResult = {
        faceDetected: false,
        faceCount: 0,
        faceRegion: null,
      }

      const formatResult = validateImageFormat(file)
      const sizeResult = validateImageSize(file)
      const faceDetectionResult = validateFaceDetection(faceResult)

      expect(formatResult.valid).toBe(true)
      expect(sizeResult.valid).toBe(true)
      expect(faceDetectionResult.valid).toBe(false)
      expect(faceDetectionResult.error?.code).toBe('FACE_NOT_DETECTED')
    })

    it('集合写真は顔検出チェックで失敗する', () => {
      const file = createMockFile('group.jpg', 'image/jpeg', 4 * 1024 * 1024)
      const faceResult: FaceDetectionResult = {
        faceDetected: true,
        faceCount: 3,
        faceRegion: { x: 100, y: 100, width: 200, height: 200 },
      }

      const formatResult = validateImageFormat(file)
      const sizeResult = validateImageSize(file)
      const faceDetectionResult = validateFaceDetection(faceResult)

      expect(formatResult.valid).toBe(true)
      expect(sizeResult.valid).toBe(true)
      expect(faceDetectionResult.valid).toBe(false)
      expect(faceDetectionResult.error?.code).toBe('MULTIPLE_FACES')
    })
  })
})

describe('エラーメッセージの日本語表示', () => {
  it('INVALID_IMAGE_FORMAT のエラーメッセージは日本語', () => {
    const file = createMockFile('test.gif', 'image/gif', 1024)
    const result = validateImageFormat(file)

    expect(result.error?.message).toMatch(/JPEG.*PNG/)
  })

  it('IMAGE_TOO_LARGE のエラーメッセージは日本語', () => {
    const file = createMockFile('test.jpg', 'image/jpeg', 15 * 1024 * 1024)
    const result = validateImageSize(file)

    expect(result.error?.message).toMatch(/10MB以下/)
  })

  it('FACE_NOT_DETECTED のエラーメッセージは日本語でガイダンスを含む', () => {
    const faceResult: FaceDetectionResult = {
      faceDetected: false,
      faceCount: 0,
      faceRegion: null,
    }
    const result = validateFaceDetection(faceResult)

    expect(result.error?.message).toMatch(/顔を検出できませんでした/)
    expect(result.error?.message).toMatch(/正面を向いた明るい写真/)
  })

  it('MULTIPLE_FACES のエラーメッセージは日本語', () => {
    const faceResult: FaceDetectionResult = {
      faceDetected: true,
      faceCount: 2,
      faceRegion: { x: 0, y: 0, width: 100, height: 100 },
    }
    const result = validateFaceDetection(faceResult)

    expect(result.error?.message).toMatch(/複数の顔が検出されました/)
    expect(result.error?.message).toMatch(/1人のみ/)
  })
})
