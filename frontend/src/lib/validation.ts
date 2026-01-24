/**
 * 画像バリデーションロジック
 *
 * 参照: functional-spec.md セクション 3.3 (SCR-002)
 */

/**
 * 画像バリデーション結果
 */
export interface ImageValidationResult {
  valid: boolean
  error?: {
    code: string
    message: string
  }
}

/**
 * 顔検出結果
 */
export interface FaceDetectionResult {
  faceDetected: boolean
  faceCount: number
  faceRegion: {
    x: number
    y: number
    width: number
    height: number
  } | null
}

/**
 * 画像形式検証
 *
 * JPEG、PNG形式のみ有効
 *
 * @param file - 検証対象のファイル
 * @returns バリデーション結果
 */
export function validateImageFormat(file: File): ImageValidationResult {
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

/**
 * 画像サイズ検証
 *
 * デフォルトは10MB以下
 *
 * @param file - 検証対象のファイル
 * @param maxSizeMB - 最大サイズ（MB単位、デフォルト: 10）
 * @returns バリデーション結果
 */
export function validateImageSize(
  file: File,
  maxSizeMB: number = 10
): ImageValidationResult {
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

/**
 * 顔検出結果検証
 *
 * - 顔が1つだけ検出されている場合のみ有効
 * - 顔が検出されない場合: FACE_NOT_DETECTED
 * - 複数の顔が検出された場合: MULTIPLE_FACES
 *
 * @param result - 顔検出結果
 * @returns バリデーション結果
 */
export function validateFaceDetection(
  result: FaceDetectionResult
): ImageValidationResult {
  // 顔が検出されない場合（faceDetected=false または faceCount=0）
  if (!result.faceDetected || result.faceCount === 0) {
    return {
      valid: false,
      error: {
        code: 'FACE_NOT_DETECTED',
        message:
          '顔を検出できませんでした。正面を向いた明るい写真をお使いください',
      },
    }
  }

  // 複数の顔が検出された場合
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
