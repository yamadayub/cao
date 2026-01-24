'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { ResultSlider } from '@/components/features/ResultSlider'
import { LoginPromptModal } from '@/components/features/LoginPromptModal'
import { ShareUrlModal } from '@/components/features/ShareUrlModal'
import { morphStages, toDataUrl } from '@/lib/api/morph'
import { createSimulation, createShareUrl } from '@/lib/api/simulations'
import { ApiError } from '@/lib/api/client'
import type { StageImage } from '@/lib/api/types'

// Clerkが利用可能かどうかを判定
const isClerkAvailable = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY

/**
 * 画像データの型
 */
interface ImageData {
  progress: number
  image: string // Base64 Data URL
}

/**
 * 状態の型
 */
interface ResultState {
  images: ImageData[]
  currentProgress: number
  isLoading: boolean
  error: string | null
  isSaving: boolean
  isSharing: boolean
  savedSimulationId: string | null
  shareUrl: string | null
}

/**
 * sessionStorageから画像データを取得
 */
function getStoredImages(): { currentImage: string | null; idealImage: string | null } {
  if (typeof window === 'undefined') {
    return { currentImage: null, idealImage: null }
  }

  return {
    currentImage: sessionStorage.getItem('cao_current_image'),
    idealImage: sessionStorage.getItem('cao_ideal_image'),
  }
}

/**
 * Data URLからBlobを作成
 */
function dataUrlToBlob(dataUrl: string): Blob {
  const arr = dataUrl.split(',')
  const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png'
  const bstr = atob(arr[1])
  let n = bstr.length
  const u8arr = new Uint8Array(n)
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n)
  }
  return new Blob([u8arr], { type: mime })
}

/**
 * Data URLからFileを作成
 */
function dataUrlToFile(dataUrl: string, filename: string): File {
  const blob = dataUrlToBlob(dataUrl)
  return new File([blob], filename, { type: blob.type })
}

/**
 * Clerk認証が有効な場合のコンポーネント
 */
function SimulationResultWithClerk() {
  // Clerkフックを条件付きでインポート（このコンポーネントはClerk有効時のみ使用）
  const { useUser, useAuth } = require('@clerk/nextjs')
  const { isSignedIn, user } = useUser()
  const { getToken } = useAuth()

  return (
    <SimulationResultContent
      isSignedIn={isSignedIn || false}
      user={user}
      getToken={getToken}
    />
  )
}

/**
 * Clerk認証が無効な場合のコンポーネント
 */
function SimulationResultWithoutClerk() {
  return (
    <SimulationResultContent
      isSignedIn={false}
      user={null}
      getToken={async () => null}
    />
  )
}

/**
 * シミュレーション結果画面の本体
 */
interface SimulationResultContentProps {
  isSignedIn: boolean
  user: { primaryEmailAddress?: { emailAddress: string } | null } | null
  getToken: () => Promise<string | null>
}

function SimulationResultContent({ isSignedIn, user, getToken }: SimulationResultContentProps) {
  const router = useRouter()

  // 状態管理
  const [state, setState] = useState<ResultState>({
    images: [],
    currentProgress: 0.5,
    isLoading: true,
    error: null,
    isSaving: false,
    isSharing: false,
    savedSimulationId: null,
    shareUrl: null,
  })

  // モーダル状態
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [showShareModal, setShowShareModal] = useState(false)
  const [loginAction, setLoginAction] = useState<'save' | 'share' | null>(null)

  // 元画像のData URL
  const [sourceImages, setSourceImages] = useState<{
    currentImage: string | null
    idealImage: string | null
  }>({ currentImage: null, idealImage: null })

  /**
   * 現在の変化度に対応する画像を取得
   */
  const currentImage = useMemo(() => {
    const image = state.images.find((img) => img.progress === state.currentProgress)
    return image?.image || null
  }, [state.images, state.currentProgress])

  /**
   * モーフィング画像を生成
   */
  const generateMorphImages = useCallback(async () => {
    const { currentImage, idealImage } = getStoredImages()

    if (!currentImage || !idealImage) {
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: '画像データが見つかりません。シミュレーション作成画面からやり直してください。',
      }))
      return
    }

    setSourceImages({ currentImage, idealImage })

    setState((prev) => ({ ...prev, isLoading: true, error: null }))

    try {
      // Data URLをFileに変換
      const currentFile = dataUrlToFile(currentImage, 'current.png')
      const idealFile = dataUrlToFile(idealImage, 'ideal.png')

      // 段階的モーフィングAPIを呼び出し
      const result = await morphStages(currentFile, idealFile)

      // 結果を状態に設定
      const images: ImageData[] = result.images.map((img: StageImage) => ({
        progress: img.progress,
        image: toDataUrl(img.image),
      }))

      setState((prev) => ({
        ...prev,
        images,
        isLoading: false,
        error: null,
      }))
    } catch (error) {
      console.error('Morphing error:', error)

      let errorMessage = '画像の生成中にエラーが発生しました。'
      if (error instanceof ApiError) {
        errorMessage = error.localizedMessage
      } else if (error instanceof Error) {
        errorMessage = error.message
      }

      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }))
    }
  }, [])

  /**
   * 初回マウント時にモーフィング画像を生成
   */
  useEffect(() => {
    generateMorphImages()
  }, [generateMorphImages])

  /**
   * スライダー値変更ハンドラ
   */
  const handleProgressChange = useCallback((value: number) => {
    setState((prev) => ({ ...prev, currentProgress: value }))
  }, [])

  /**
   * 保存ボタンクリックハンドラ
   */
  const handleSave = useCallback(async () => {
    if (!isSignedIn) {
      setLoginAction('save')
      setShowLoginModal(true)
      return
    }

    if (state.savedSimulationId) {
      // 既に保存済み
      return
    }

    setState((prev) => ({ ...prev, isSaving: true }))

    try {
      const token = await getToken()
      if (!token) {
        throw new Error('認証トークンを取得できませんでした')
      }

      const result = await createSimulation(
        {
          current_image: sourceImages.currentImage || '',
          ideal_image: sourceImages.idealImage || '',
          result_images: state.images.map((img) => ({
            progress: img.progress,
            image: img.image,
          })),
          settings: {
            selected_progress: state.currentProgress,
          },
        },
        token
      )

      setState((prev) => ({
        ...prev,
        isSaving: false,
        savedSimulationId: result.id,
      }))
    } catch (error) {
      console.error('Save error:', error)

      let errorMessage = '保存中にエラーが発生しました。'
      if (error instanceof ApiError) {
        errorMessage = error.localizedMessage
      }

      setState((prev) => ({
        ...prev,
        isSaving: false,
        error: errorMessage,
      }))
    }
  }, [isSignedIn, getToken, sourceImages, state.images, state.currentProgress, state.savedSimulationId])

  /**
   * 共有URLボタンクリックハンドラ
   */
  const handleShare = useCallback(async () => {
    if (!isSignedIn) {
      setLoginAction('share')
      setShowLoginModal(true)
      return
    }

    setState((prev) => ({ ...prev, isSharing: true }))

    try {
      const token = await getToken()
      if (!token) {
        throw new Error('認証トークンを取得できませんでした')
      }

      let simulationId = state.savedSimulationId

      // 未保存の場合は先に保存
      if (!simulationId) {
        const saveResult = await createSimulation(
          {
            current_image: sourceImages.currentImage || '',
            ideal_image: sourceImages.idealImage || '',
            result_images: state.images.map((img) => ({
              progress: img.progress,
              image: img.image,
            })),
            settings: {
              selected_progress: state.currentProgress,
            },
          },
          token
        )
        simulationId = saveResult.id
        setState((prev) => ({ ...prev, savedSimulationId: simulationId }))
      }

      // 共有URL生成
      const shareResult = await createShareUrl(simulationId, token)

      setState((prev) => ({
        ...prev,
        isSharing: false,
        shareUrl: shareResult.share_url,
      }))

      setShowShareModal(true)
    } catch (error) {
      console.error('Share error:', error)

      let errorMessage = '共有URL生成中にエラーが発生しました。'
      if (error instanceof ApiError) {
        errorMessage = error.localizedMessage
      }

      setState((prev) => ({
        ...prev,
        isSharing: false,
        error: errorMessage,
      }))
    }
  }, [isSignedIn, getToken, sourceImages, state.images, state.currentProgress, state.savedSimulationId])

  /**
   * 新規作成ボタンクリックハンドラ
   */
  const handleNewSimulation = useCallback(() => {
    // sessionStorageをクリア
    sessionStorage.removeItem('cao_current_image')
    sessionStorage.removeItem('cao_ideal_image')
    router.push('/simulate')
  }, [router])

  /**
   * ログインボタンクリックハンドラ
   */
  const handleLogin = useCallback(() => {
    setShowLoginModal(false)
    router.push('/sign-in')
  }, [router])

  /**
   * 再試行ハンドラ
   */
  const handleRetry = useCallback(() => {
    generateMorphImages()
  }, [generateMorphImages])

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900">Cao</h1>
          <nav className="flex gap-4">
            {isSignedIn ? (
              <>
                <span className="text-sm text-gray-600">
                  {user?.primaryEmailAddress?.emailAddress}
                </span>
                <a
                  href="/mypage"
                  className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  マイページ
                </a>
              </>
            ) : (
              <>
                <a
                  href="/sign-in"
                  className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  ログイン
                </a>
                <a
                  href="/mypage"
                  className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  マイページ
                </a>
              </>
            )}
          </nav>
        </div>
      </header>

      {/* メインコンテンツ */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* ページタイトル */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900">シミュレーション結果</h2>
        </div>

        {/* エラー表示 */}
        {state.error && (
          <div
            className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg"
            role="alert"
            data-testid="error-message"
          >
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div className="flex-1">
                <p className="text-red-700">{state.error}</p>
                <button
                  type="button"
                  onClick={handleRetry}
                  className="mt-2 text-sm font-medium text-red-600 hover:text-red-800 underline"
                  data-testid="retry-button"
                >
                  再試行
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ローディング表示 */}
        {state.isLoading && (
          <div
            className="flex flex-col items-center justify-center py-16"
            data-testid="loading"
          >
            <svg
              className="animate-spin h-12 w-12 text-blue-600 mb-4"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              aria-hidden="true"
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
            <p className="text-gray-600 text-lg">シミュレーション画像を生成中...</p>
            <p className="text-gray-500 text-sm mt-2">しばらくお待ちください</p>
          </div>
        )}

        {/* 結果表示 */}
        {!state.isLoading && !state.error && state.images.length > 0 && (
          <>
            {/* 結果画像 */}
            <div
              className="bg-white rounded-xl shadow-sm p-4 mb-8"
              data-testid="result-image-container"
            >
              <div className="aspect-square max-w-lg mx-auto overflow-hidden rounded-lg bg-gray-100">
                {currentImage ? (
                  <img
                    src={currentImage}
                    alt={`変化度${Math.round(state.currentProgress * 100)}%のシミュレーション結果`}
                    className="w-full h-full object-cover"
                    data-testid="result-image"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-400">
                    画像を読み込み中...
                  </div>
                )}
              </div>
            </div>

            {/* スライダー */}
            <div className="mb-8">
              <ResultSlider
                value={state.currentProgress}
                onChange={handleProgressChange}
                disabled={state.isSaving || state.isSharing}
                testId="result-slider"
              />
            </div>

            {/* 保存・共有ボタン */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-6">
              <button
                type="button"
                onClick={handleSave}
                disabled={state.isSaving || !!state.savedSimulationId}
                className={`
                  px-8 py-3 text-base font-semibold rounded-lg
                  focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all
                  ${
                    state.savedSimulationId
                      ? 'bg-green-600 text-white cursor-default'
                      : state.isSaving
                        ? 'bg-gray-400 text-white cursor-wait'
                        : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
                  }
                `}
                data-testid="save-button"
              >
                {state.isSaving ? (
                  <span className="flex items-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
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
                    保存中...
                  </span>
                ) : state.savedSimulationId ? (
                  <span className="flex items-center gap-2">
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    保存済み
                  </span>
                ) : (
                  '保存'
                )}
              </button>

              <button
                type="button"
                onClick={handleShare}
                disabled={state.isSharing}
                className={`
                  px-8 py-3 text-base font-semibold rounded-lg
                  focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all
                  ${
                    state.isSharing
                      ? 'bg-gray-400 text-white cursor-wait'
                      : 'bg-white text-blue-600 border-2 border-blue-600 hover:bg-blue-50 focus:ring-blue-500'
                  }
                `}
                data-testid="share-button"
              >
                {state.isSharing ? (
                  <span className="flex items-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
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
                    生成中...
                  </span>
                ) : (
                  '共有URL'
                )}
              </button>
            </div>

            {/* 新規作成ボタン */}
            <div className="flex justify-center">
              <button
                type="button"
                onClick={handleNewSimulation}
                className="w-full max-w-md px-8 py-4 text-lg font-semibold text-gray-700 bg-white border-2 border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 transition-all"
                data-testid="new-simulation-button"
              >
                新しいシミュレーションを作成
              </button>
            </div>
          </>
        )}

        {/* 画像がない場合のエラー状態 */}
        {!state.isLoading && !state.error && state.images.length === 0 && (
          <div className="text-center py-16">
            <svg
              className="mx-auto h-12 w-12 text-gray-400 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <p className="text-gray-600 mb-4">
              シミュレーション結果がありません
            </p>
            <button
              type="button"
              onClick={handleNewSimulation}
              className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
              data-testid="new-simulation-button"
            >
              シミュレーションを作成する
            </button>
          </div>
        )}
      </div>

      {/* ログイン誘導モーダル */}
      <LoginPromptModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onLogin={handleLogin}
        title={
          loginAction === 'save'
            ? '保存するにはログインが必要です'
            : '共有するにはログインが必要です'
        }
        description={
          loginAction === 'save'
            ? 'シミュレーション結果を保存するにはログインが必要です。'
            : 'シミュレーション結果を共有するにはログインが必要です。'
        }
        testId="login-prompt-modal"
      />

      {/* 共有URLモーダル */}
      <ShareUrlModal
        isOpen={showShareModal}
        shareUrl={state.shareUrl || ''}
        onClose={() => setShowShareModal(false)}
        testId="share-url-modal"
      />
    </main>
  )
}

/**
 * シミュレーション結果画面 (SCR-003)
 *
 * 参照: functional-spec.md セクション 3.4
 * 参照: business-spec.md UC-004, UC-005, UC-006, UC-007
 */
export default function SimulationResultPage() {
  // Clerkが利用可能かどうかでコンポーネントを切り替え
  if (isClerkAvailable) {
    return <SimulationResultWithClerk />
  }
  return <SimulationResultWithoutClerk />
}
