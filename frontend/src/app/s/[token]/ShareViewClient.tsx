'use client'

import { useCallback, useEffect, useState, useMemo } from 'react'
import Link from 'next/link'
import { getSharedSimulation } from '@/lib/api/simulations'
import { ResultImage } from '@/lib/api/types'
import { ApiError } from '@/lib/api/client'
import { ResultSlider } from '@/components/features/ResultSlider'

interface ShareViewClientProps {
  /** 共有トークン */
  token: string
  /** テスト用のdata-testid */
  testId?: string
}

export function ShareViewClient({ token, testId }: ShareViewClientProps) {
  // State
  const [resultImages, setResultImages] = useState<ResultImage[]>([])
  const [createdAt, setCreatedAt] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0.5)

  /**
   * 共有シミュレーションを取得
   */
  useEffect(() => {
    const fetchSharedSimulation = async () => {
      try {
        setIsLoading(true)
        setError(null)

        const data = await getSharedSimulation(token)
        setResultImages(data.result_images)
        setCreatedAt(data.created_at)
      } catch (err) {
        console.error('Failed to fetch shared simulation:', err)
        if (err instanceof ApiError) {
          if (err.code === 'NOT_FOUND') {
            setError('このURLは無効または期限切れです')
          } else {
            setError(err.localizedMessage)
          }
        } else {
          setError('シミュレーションの取得に失敗しました')
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchSharedSimulation()
  }, [token])

  /**
   * 現在の進捗に対応する画像を取得
   */
  const currentImage = useMemo(() => {
    if (resultImages.length === 0) return null

    // 最も近い進捗の画像を見つける
    let closestImage = resultImages[0]
    let minDistance = Math.abs(progress - closestImage.progress)

    for (const img of resultImages) {
      const distance = Math.abs(progress - img.progress)
      if (distance < minDistance) {
        minDistance = distance
        closestImage = img
      }
    }

    return closestImage
  }, [resultImages, progress])

  /**
   * スライダー変更ハンドラ
   */
  const handleProgressChange = useCallback((newProgress: number) => {
    setProgress(newProgress)
  }, [])

  // エラー画面
  if (error && !isLoading) {
    return (
      <div className="min-h-screen bg-gray-50" data-testid={testId}>
        {/* ヘッダー */}
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <Link href="/" className="text-2xl font-bold text-blue-600">
              Cao
            </Link>
          </div>
        </header>

        <main className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            {/* 警告アイコン */}
            <div className="mx-auto w-16 h-16 flex items-center justify-center bg-yellow-100 rounded-full mb-6">
              <svg
                className="w-8 h-8 text-yellow-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>

            <h1
              className="text-2xl font-bold text-gray-900 mb-4"
              data-testid={testId ? `${testId}-error-title` : undefined}
            >
              {error}
            </h1>

            <Link
              href="/"
              className="inline-block px-8 py-3 text-base font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
              data-testid={testId ? `${testId}-back-home` : undefined}
            >
              トップページへ戻る
            </Link>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50" data-testid={testId}>
      {/* ヘッダー */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Link href="/" className="text-2xl font-bold text-blue-600">
            Cao
          </Link>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-2xl font-bold text-gray-900 text-center mb-8">
          共有されたシミュレーション
        </h1>

        {/* ローディング */}
        {isLoading && (
          <div className="flex justify-center items-center py-24">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
          </div>
        )}

        {/* コンテンツ */}
        {!isLoading && !error && (
          <>
            {/* 結果画像 */}
            <div className="mb-8">
              <div
                className="bg-white rounded-xl shadow-lg overflow-hidden"
                data-testid={testId ? `${testId}-result-image-container` : undefined}
              >
                <div className="aspect-square bg-gray-100 flex items-center justify-center">
                  {currentImage ? (
                    <img
                      src={currentImage.url}
                      alt={`シミュレーション結果 ${Math.round(currentImage.progress * 100)}%`}
                      className="w-full h-full object-cover"
                      data-testid={testId ? `${testId}-result-image` : undefined}
                    />
                  ) : (
                    <div className="text-gray-400">
                      <svg
                        className="w-16 h-16"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                        />
                      </svg>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* スライダー */}
            <div className="mb-8">
              <ResultSlider
                value={progress}
                onChange={handleProgressChange}
                testId={testId ? `${testId}-slider` : undefined}
              />
            </div>

            {/* CTAボタン */}
            <div className="text-center">
              <Link
                href="/simulate"
                className="inline-block w-full max-w-md px-8 py-4 text-lg font-semibold text-white bg-blue-600 rounded-xl hover:bg-blue-700 transition-colors shadow-lg"
                data-testid={testId ? `${testId}-try-simulation` : undefined}
              >
                自分もシミュレーションを試す
              </Link>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
