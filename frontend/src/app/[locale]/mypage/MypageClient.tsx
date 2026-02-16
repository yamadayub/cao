'use client'

import { useCallback, useEffect, useState } from 'react'
import { useAuth, useUser, useClerk } from '@clerk/nextjs'
import { useTranslations, useLocale } from 'next-intl'
import { useRouter } from '@/i18n/navigation'
import { Link } from '@/i18n/navigation'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { getSimulations, deleteSimulation, createShareUrl } from '@/lib/api/simulations'
import { SimulationSummary, Pagination } from '@/lib/api/types'
import { ApiError } from '@/lib/api/client'
import { DeleteConfirmModal } from '@/components/features/DeleteConfirmModal'
import { ShareUrlModal } from '@/components/features/ShareUrlModal'

const PAGE_SIZE = 20

interface MypageClientProps {
  /** テスト用のdata-testid */
  testId?: string
}

export function MypageClient({ testId }: MypageClientProps) {
  const router = useRouter()
  const { getToken, isSignedIn } = useAuth()
  const { user } = useUser()
  const { signOut } = useClerk()
  const t = useTranslations('mypage')
  const locale = useLocale()

  // State
  const [simulations, setSimulations] = useState<SimulationSummary[]>([])
  const [pagination, setPagination] = useState<Pagination | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // モーダル状態
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const [shareUrl, setShareUrl] = useState<string | null>(null)
  const [isGeneratingShareUrl, setIsGeneratingShareUrl] = useState(false)

  /**
   * シミュレーション一覧を取得
   */
  const fetchSimulations = useCallback(async (offset: number = 0, append: boolean = false) => {
    if (!isSignedIn) return

    try {
      if (append) {
        setIsLoadingMore(true)
      } else {
        setIsLoading(true)
      }
      setError(null)

      const token = await getToken()
      if (!token) {
        throw new Error(t('errors.authTokenFailed'))
      }

      const data = await getSimulations(token, {
        limit: PAGE_SIZE,
        offset,
      })

      if (append) {
        setSimulations(prev => [...prev, ...data.simulations])
      } else {
        setSimulations(data.simulations)
      }
      setPagination(data.pagination)
    } catch (err) {
      console.error('Failed to fetch simulations:', err)
      if (err instanceof ApiError) {
        setError(err.localizedMessage)
      } else if (err instanceof Error) {
        setError(err.message)
      } else {
        setError(t('errors.fetchFailed'))
      }
    } finally {
      setIsLoading(false)
      setIsLoadingMore(false)
    }
  }, [isSignedIn, getToken, t])

  /**
   * 初回ロード
   */
  useEffect(() => {
    if (isSignedIn) {
      fetchSimulations()
    }
  }, [isSignedIn, fetchSimulations])

  /**
   * さらに読み込む
   */
  const handleLoadMore = useCallback(() => {
    if (!pagination || !pagination.has_more || isLoadingMore) return
    fetchSimulations(pagination.offset + pagination.limit, true)
  }, [pagination, isLoadingMore, fetchSimulations])

  /**
   * サムネイルクリック - 詳細画面へ遷移
   */
  const handleThumbnailClick = useCallback((id: string) => {
    router.push(`/simulate/result?id=${id}`)
  }, [router])

  /**
   * 削除モーダルを開く
   */
  const handleDeleteClick = useCallback((id: string) => {
    setDeleteTargetId(id)
  }, [])

  /**
   * 削除確定
   */
  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTargetId || !isSignedIn) return

    try {
      setIsDeleting(true)
      const token = await getToken()
      if (!token) {
        throw new Error(t('errors.authTokenFailed'))
      }

      await deleteSimulation(deleteTargetId, token)

      // ローカルリストから削除
      setSimulations(prev => prev.filter(sim => sim.id !== deleteTargetId))
      if (pagination) {
        setPagination(prev => prev ? { ...prev, total: prev.total - 1 } : null)
      }

      setDeleteTargetId(null)
    } catch (err) {
      console.error('Failed to delete simulation:', err)
      if (err instanceof ApiError) {
        alert(err.localizedMessage)
      } else {
        alert(t('errors.deleteFailed'))
      }
    } finally {
      setIsDeleting(false)
    }
  }, [deleteTargetId, isSignedIn, getToken, pagination, t])

  /**
   * 削除キャンセル
   */
  const handleDeleteCancel = useCallback(() => {
    if (!isDeleting) {
      setDeleteTargetId(null)
    }
  }, [isDeleting])

  /**
   * 共有URLを生成
   */
  const handleShareClick = useCallback(async (id: string) => {
    if (!isSignedIn) return

    try {
      setIsGeneratingShareUrl(true)
      const token = await getToken()
      if (!token) {
        throw new Error(t('errors.authTokenFailed'))
      }

      const data = await createShareUrl(id, token)
      setShareUrl(data.share_url)

      // 対象シミュレーションをis_public: trueに更新（ローカル）
      setSimulations(prev =>
        prev.map(sim =>
          sim.id === id ? { ...sim, is_public: true } : sim
        )
      )
    } catch (err) {
      console.error('Failed to generate share URL:', err)
      if (err instanceof ApiError) {
        alert(err.localizedMessage)
      } else {
        alert(t('errors.shareFailed'))
      }
    } finally {
      setIsGeneratingShareUrl(false)
    }
  }, [isSignedIn, getToken, t])

  /**
   * 共有モーダルを閉じる
   */
  const handleShareModalClose = useCallback(() => {
    setShareUrl(null)
  }, [])

  /**
   * 日付フォーマット
   */
  const formatDate = useCallback((dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString(locale, {
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
    })
  }, [locale])

  // ユーザーのメールアドレス
  const userEmail = user?.primaryEmailAddress?.emailAddress || ''

  return (
    <div className="min-h-screen flex flex-col bg-neutral-50" data-testid={testId}>
      <Header />

      <main className="flex-1 pt-20">
        <div className="container-narrow py-8 md:py-12">
          {/* ページタイトル */}
          <div className="text-center mb-8 md:mb-12">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">My Page</p>
            <h1 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900 mb-4">
              {t('title')}
            </h1>
            <div className="inline-block bg-white rounded-xl shadow-elegant px-6 py-3">
              <p
                className="text-neutral-700 mb-3"
                data-testid={testId ? `${testId}-user-email` : undefined}
              >
                {userEmail}
              </p>
              <button
                type="button"
                onClick={() => signOut({ redirectUrl: '/' })}
                className="text-sm text-neutral-500 hover:text-neutral-700 underline transition-colors"
                data-testid={testId ? `${testId}-sign-out` : undefined}
              >
                {t('signOut')}
              </button>
            </div>
          </div>

          {/* 新規作成ボタン */}
          <div className="mb-8 flex justify-center">
            <Link
              href="/simulate"
              className="px-8 py-3 text-base font-medium text-white bg-primary-700 rounded-full hover:bg-primary-800 hover:shadow-elegant transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
              data-testid={testId ? `${testId}-new-simulation` : undefined}
            >
              {t('newSimulation')}
            </Link>
          </div>

          {/* シミュレーション一覧 */}
          <div>
            <h2 className="font-serif text-xl text-neutral-800 text-center mb-6">
              {t('savedSimulations')}
              {pagination && (
                <span className="ml-2 text-neutral-500 font-normal">
                  {t('count', { count: pagination.total })}
                </span>
              )}
            </h2>

            {/* ローディング */}
            {isLoading && (
              <div className="flex justify-center items-center py-12">
                <div className="w-10 h-10 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin" />
              </div>
            )}

            {/* エラー */}
            {error && !isLoading && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
                {error}
                <button
                  onClick={() => fetchSimulations()}
                  className="ml-4 text-primary-600 underline hover:text-primary-800"
                >
                  {t('retry')}
                </button>
              </div>
            )}

            {/* シミュレーション一覧 */}
            {!isLoading && !error && (
              <>
                {simulations.length === 0 ? (
                  <div className="text-center py-12 bg-white rounded-2xl shadow-elegant">
                    <p className="text-neutral-500 mb-6">
                      {t('noSimulations')}
                    </p>
                    <Link
                      href="/simulate"
                      className="inline-block px-6 py-3 text-sm font-medium text-white bg-primary-700 rounded-full hover:bg-primary-800 hover:shadow-elegant transition-all duration-300"
                    >
                      {t('createSimulation')}
                    </Link>
                  </div>
                ) : (
                  <>
                    <div
                      className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
                      data-testid={testId ? `${testId}-simulation-grid` : undefined}
                    >
                      {simulations.map((sim) => (
                        <div
                          key={sim.id}
                          className="bg-white rounded-2xl shadow-elegant overflow-hidden"
                          data-testid={testId ? `${testId}-simulation-card-${sim.id}` : undefined}
                        >
                          {/* サムネイル */}
                          <button
                            type="button"
                            onClick={() => handleThumbnailClick(sim.id)}
                            className="w-full aspect-square bg-neutral-100 overflow-hidden focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-inset"
                            aria-label={t('viewSimulation', { date: formatDate(sim.created_at) })}
                          >
                            {sim.thumbnail_url ? (
                              <img
                                src={sim.thumbnail_url}
                                alt={t('simulationAlt', { date: formatDate(sim.created_at) })}
                                className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                              />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-neutral-400">
                                <svg
                                  className="w-12 h-12"
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
                          </button>

                          {/* カード情報 */}
                          <div className="p-4">
                            <p className="text-sm text-neutral-600 mb-3">
                              {formatDate(sim.created_at)}
                            </p>
                            <div className="flex gap-2">
                              <button
                                type="button"
                                onClick={() => handleShareClick(sim.id)}
                                disabled={isGeneratingShareUrl}
                                className="flex-1 px-3 py-2 text-sm font-medium text-primary-700 bg-primary-50 rounded-full hover:bg-primary-100 transition-all duration-300 disabled:opacity-50"
                                data-testid={testId ? `${testId}-share-button-${sim.id}` : undefined}
                              >
                                {t('share')}
                              </button>
                              <button
                                type="button"
                                onClick={() => handleDeleteClick(sim.id)}
                                className="flex-1 px-3 py-2 text-sm font-medium text-red-600 bg-red-50 rounded-full hover:bg-red-100 transition-all duration-300"
                                data-testid={testId ? `${testId}-delete-button-${sim.id}` : undefined}
                              >
                                {t('delete')}
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* さらに読み込むボタン */}
                    {pagination?.has_more && (
                      <div className="mt-8 text-center">
                        <button
                          type="button"
                          onClick={handleLoadMore}
                          disabled={isLoadingMore}
                          className="px-8 py-3 text-base font-medium text-primary-700 bg-white border border-primary-300 rounded-full hover:bg-primary-50 hover:border-primary-400 transition-all duration-300 disabled:opacity-50"
                          data-testid={testId ? `${testId}-load-more` : undefined}
                        >
                          {isLoadingMore ? (
                            <span className="flex items-center justify-center gap-2">
                              <svg
                                className="w-5 h-5 animate-spin"
                                fill="none"
                                viewBox="0 0 24 24"
                              >
                                <circle
                                  className="opacity-25"
                                  cx="12"
                                  cy="12"
                                  r="10"
                                  stroke="currentColor"
                                  strokeWidth="4"
                                />
                                <path
                                  className="opacity-75"
                                  fill="currentColor"
                                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                />
                              </svg>
                              {t('loadingMore')}
                            </span>
                          ) : (
                            t('loadMore')
                          )}
                        </button>
                      </div>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </main>

      <Footer />

      {/* 削除確認モーダル */}
      <DeleteConfirmModal
        isOpen={deleteTargetId !== null}
        targetId={deleteTargetId || undefined}
        isDeleting={isDeleting}
        onConfirm={handleDeleteConfirm}
        onCancel={handleDeleteCancel}
        testId={testId ? `${testId}-delete-modal` : undefined}
      />

      {/* 共有URLモーダル */}
      <ShareUrlModal
        isOpen={shareUrl !== null}
        shareUrl={shareUrl || ''}
        onClose={handleShareModalClose}
        testId={testId ? `${testId}-share-modal` : undefined}
      />
    </div>
  )
}
