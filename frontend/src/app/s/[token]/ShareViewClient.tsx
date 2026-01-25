'use client'

import { useCallback, useEffect, useState, useMemo } from 'react'
import Link from 'next/link'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
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
      <div className="min-h-screen flex flex-col bg-neutral-50" data-testid={testId}>
        <Header />

        <main className="flex-1 flex flex-col items-center justify-center pt-20 pb-12 px-4">
          <div className="text-center">
            {/* 警告アイコン */}
            <div className="mx-auto w-16 h-16 flex items-center justify-center bg-amber-50 rounded-full mb-6">
              <svg
                className="w-8 h-8 text-amber-600"
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
              className="font-serif text-2xl text-neutral-900 mb-6"
              data-testid={testId ? `${testId}-error-title` : undefined}
            >
              {error}
            </h1>

            <Link
              href="/"
              className="inline-block px-8 py-3 text-base font-medium text-white bg-primary-700 rounded-full hover:bg-primary-800 hover:shadow-elegant transition-all duration-300"
              data-testid={testId ? `${testId}-back-home` : undefined}
            >
              トップページへ戻る
            </Link>
          </div>
        </main>

        <Footer />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-neutral-50" data-testid={testId}>
      <Header />

      <main className="flex-1 pt-20">
        <div className="container-narrow py-8 md:py-12">
          {/* ページタイトル */}
          <div className="text-center mb-8 md:mb-12">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">Shared</p>
            <h1 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900">
              共有されたシミュレーション
            </h1>
          </div>

          {/* ローディング */}
          {isLoading && (
            <div className="flex flex-col items-center justify-center py-16">
              <div className="w-12 h-12 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin mb-4"></div>
              <p className="text-neutral-700 text-base font-serif">読み込み中...</p>
            </div>
          )}

          {/* コンテンツ */}
          {!isLoading && !error && (
            <>
              {/* 結果画像 */}
              <div className="mb-8">
                <div
                  className="bg-white rounded-2xl shadow-elegant overflow-hidden"
                  data-testid={testId ? `${testId}-result-image-container` : undefined}
                >
                  <div className="aspect-square bg-neutral-100 flex items-center justify-center">
                    {currentImage ? (
                      <img
                        src={currentImage.url}
                        alt={`シミュレーション結果 ${Math.round(currentImage.progress * 100)}%`}
                        className="w-full h-full object-cover"
                        data-testid={testId ? `${testId}-result-image` : undefined}
                      />
                    ) : (
                      <div className="text-neutral-400">
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
                  className="inline-block w-full max-w-md px-8 py-4 text-base font-medium text-white bg-primary-700 rounded-full hover:bg-primary-800 hover:shadow-elegant transition-all duration-300"
                  data-testid={testId ? `${testId}-try-simulation` : undefined}
                >
                  自分もシミュレーションを試す
                </Link>
              </div>
            </>
          )}
        </div>
      </main>

      <Footer />
    </div>
  )
}
