'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

export interface CameraCaptureProps {
  /** ガイダンス表示用の理想の顔画像URL（Data URL） */
  guideImageUrl: string
  /** 撮影完了時のコールバック */
  onCapture: (file: File, previewUrl: string) => void
  /** キャンセル時のコールバック */
  onCancel: () => void
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * カメラキャプチャコンポーネント
 *
 * スマートフォンのカメラを起動し、理想の顔画像をガイドとして表示しながら
 * 現在の顔を撮影する
 */
export function CameraCapture({
  guideImageUrl,
  onCapture,
  onCancel,
  testId,
}: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isCaptured, setIsCaptured] = useState(false)
  const [capturedImageUrl, setCapturedImageUrl] = useState<string | null>(null)
  const [guideOpacity, setGuideOpacity] = useState(0.3)

  /**
   * カメラを起動
   */
  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true)
      setError(null)

      // フロントカメラを優先して使用
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 1280 },
        },
        audio: false,
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      setIsLoading(false)
    } catch (err) {
      console.error('カメラ起動エラー:', err)
      setIsLoading(false)

      if (err instanceof Error) {
        if (err.name === 'NotAllowedError') {
          setError('カメラへのアクセスが許可されていません。ブラウザの設定でカメラアクセスを許可してください。')
        } else if (err.name === 'NotFoundError') {
          setError('カメラが見つかりません。カメラが接続されているか確認してください。')
        } else {
          setError('カメラを起動できませんでした。')
        }
      } else {
        setError('カメラを起動できませんでした。')
      }
    }
  }, [])

  /**
   * カメラを停止
   */
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
  }, [])

  /**
   * 写真を撮影
   */
  const handleCapture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // ビデオのサイズを取得
    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight

    // 正方形にクロップ（中央部分を使用）
    const size = Math.min(videoWidth, videoHeight)
    const offsetX = (videoWidth - size) / 2
    const offsetY = (videoHeight - size) / 2

    // キャンバスサイズを設定
    canvas.width = size
    canvas.height = size

    // フロントカメラの場合は左右反転
    ctx.save()
    ctx.scale(-1, 1)
    ctx.drawImage(
      video,
      offsetX, offsetY, size, size,
      -size, 0, size, size
    )
    ctx.restore()

    // Data URLとして取得
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9)
    setCapturedImageUrl(dataUrl)
    setIsCaptured(true)

    // カメラを一時停止
    video.pause()
  }, [])

  /**
   * 撮り直し
   */
  const handleRetake = useCallback(async () => {
    setIsCaptured(false)
    setCapturedImageUrl(null)

    if (videoRef.current) {
      await videoRef.current.play()
    }
  }, [])

  /**
   * 撮影した画像を使用
   */
  const handleUsePhoto = useCallback(() => {
    if (!capturedImageUrl || !canvasRef.current) return

    // Data URLをBlobに変換
    canvasRef.current.toBlob(
      (blob) => {
        if (blob) {
          const file = new File([blob], 'captured-face.jpg', { type: 'image/jpeg' })
          onCapture(file, capturedImageUrl)
        }
      },
      'image/jpeg',
      0.9
    )
  }, [capturedImageUrl, onCapture])

  /**
   * ガイド透明度を調整
   */
  const handleGuideOpacityChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setGuideOpacity(parseFloat(e.target.value))
  }, [])

  // コンポーネントマウント時にカメラ起動
  useEffect(() => {
    startCamera()
    return () => {
      stopCamera()
    }
  }, [startCamera, stopCamera])

  return (
    <div
      className="fixed inset-0 z-50 bg-black flex flex-col"
      data-testid={testId}
    >
      {/* ヘッダー */}
      <div className="flex items-center justify-between p-4 bg-black/80">
        <button
          type="button"
          onClick={onCancel}
          className="text-white p-2 hover:bg-white/10 rounded-full transition-colors"
          aria-label="閉じる"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <h2 className="text-white font-medium">現在の顔を撮影</h2>
        <div className="w-10" /> {/* スペーサー */}
      </div>

      {/* メインエリア */}
      <div className="flex-1 relative overflow-hidden">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black">
            <div className="text-center text-white">
              <div className="w-12 h-12 border-2 border-white/30 border-t-white rounded-full animate-spin mx-auto mb-4" />
              <p>カメラを起動中...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black p-6">
            <div className="text-center text-white max-w-md">
              <svg className="w-16 h-16 text-red-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <p className="mb-6">{error}</p>
              <button
                type="button"
                onClick={onCancel}
                className="px-6 py-3 bg-white text-black rounded-full font-medium"
              >
                戻る
              </button>
            </div>
          </div>
        )}

        {!isLoading && !error && (
          <>
            {/* カメラプレビュー */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="relative w-full max-w-lg aspect-square">
                {/* ビデオ（左右反転） */}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`absolute inset-0 w-full h-full object-cover ${isCaptured ? 'hidden' : ''}`}
                  style={{ transform: 'scaleX(-1)' }}
                />

                {/* 撮影済み画像 */}
                {isCaptured && capturedImageUrl && (
                  <img
                    src={capturedImageUrl}
                    alt="撮影した画像"
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                )}

                {/* ガイドオーバーレイ（理想の顔） */}
                {!isCaptured && (
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{ opacity: guideOpacity }}
                  >
                    <img
                      src={guideImageUrl}
                      alt="ガイド"
                      className="w-full h-full object-cover"
                    />
                  </div>
                )}

                {/* 顔の位置ガイド（楕円形の枠） */}
                {!isCaptured && (
                  <div className="absolute inset-0 pointer-events-none">
                    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                      {/* 外側の暗い部分 */}
                      <defs>
                        <mask id="faceMask">
                          <rect x="0" y="0" width="100" height="100" fill="white" />
                          <ellipse cx="50" cy="45" rx="28" ry="38" fill="black" />
                        </mask>
                      </defs>
                      <rect x="0" y="0" width="100" height="100" fill="rgba(0,0,0,0.4)" mask="url(#faceMask)" />

                      {/* 顔の枠線 */}
                      <ellipse
                        cx="50" cy="45" rx="28" ry="38"
                        fill="none"
                        stroke="white"
                        strokeWidth="0.5"
                        strokeDasharray="2,2"
                      />
                    </svg>
                  </div>
                )}

                {/* 隠しキャンバス（撮影用） */}
                <canvas ref={canvasRef} className="hidden" />
              </div>
            </div>

            {/* ガイダンステキスト */}
            {!isCaptured && (
              <div className="absolute top-4 left-0 right-0 text-center px-4">
                <p className="text-white text-sm bg-black/50 inline-block px-4 py-2 rounded-full">
                  理想の顔と同じ向き・大きさで撮影してください
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* コントロールエリア */}
      {!isLoading && !error && (
        <div className="bg-black/80 p-6 space-y-4">
          {/* ガイド透明度スライダー（撮影前のみ） */}
          {!isCaptured && (
            <div className="flex items-center gap-4 px-4">
              <span className="text-white/70 text-sm whitespace-nowrap">ガイド透明度</span>
              <input
                type="range"
                min="0"
                max="0.7"
                step="0.1"
                value={guideOpacity}
                onChange={handleGuideOpacityChange}
                className="flex-1 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-white/70 text-sm w-10 text-right">
                {Math.round(guideOpacity * 100)}%
              </span>
            </div>
          )}

          {/* ボタン */}
          <div className="flex items-center justify-center gap-6">
            {!isCaptured ? (
              /* 撮影ボタン */
              <button
                type="button"
                onClick={handleCapture}
                className="w-20 h-20 rounded-full bg-white flex items-center justify-center hover:bg-gray-100 transition-colors focus:outline-none focus:ring-4 focus:ring-white/50"
                aria-label="撮影"
                data-testid={testId ? `${testId}-capture-button` : undefined}
              >
                <div className="w-16 h-16 rounded-full border-4 border-black/20" />
              </button>
            ) : (
              /* 撮影後のボタン */
              <>
                <button
                  type="button"
                  onClick={handleRetake}
                  className="px-6 py-3 text-white border border-white rounded-full font-medium hover:bg-white/10 transition-colors"
                  data-testid={testId ? `${testId}-retake-button` : undefined}
                >
                  撮り直す
                </button>
                <button
                  type="button"
                  onClick={handleUsePhoto}
                  className="px-6 py-3 bg-primary-600 text-white rounded-full font-medium hover:bg-primary-700 transition-colors"
                  data-testid={testId ? `${testId}-use-button` : undefined}
                >
                  この写真を使う
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
