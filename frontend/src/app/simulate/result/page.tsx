'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { ResultSlider } from '@/components/features/ResultSlider'
import { LoginPromptModal } from '@/components/features/LoginPromptModal'
import { ShareUrlModal } from '@/components/features/ShareUrlModal'
import { morphStages, blendParts, toDataUrl } from '@/lib/api/morph'
import { createSimulation, createShareUrl } from '@/lib/api/simulations'
import { ApiError } from '@/lib/api/client'
import type { StageImage, PartsSelection } from '@/lib/api/types'
import { PARTS_DISPLAY_NAMES } from '@/lib/api/types'
import { PartsSelector } from '@/components/features/PartsSelector'

// Clerkの型定義
interface ClerkInstance {
  user?: { primaryEmailAddress?: { emailAddress: string } | null } | null
  session?: { getToken: () => Promise<string | null> } | null
}

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
 * パーツブレンド状態の型
 */
interface PartsBlendState {
  selection: PartsSelection
  image: string | null
  isLoading: boolean
  error: string | null
}

/**
 * デフォルトのパーツ選択（全てOFF）
 */
const defaultPartsSelection: PartsSelection = {
  left_eye: false,
  right_eye: false,
  left_eyebrow: false,
  right_eyebrow: false,
  nose: false,
  lips: false,
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
 * Clerkの状態を安全に取得するカスタムフック
 */
function useClerkState() {
  const [clerkState, setClerkState] = useState<{
    isSignedIn: boolean
    user: { primaryEmailAddress?: { emailAddress: string } | null } | null
    getToken: () => Promise<string | null>
  }>({
    isSignedIn: false,
    user: null,
    getToken: async () => null,
  })

  useEffect(() => {
    const checkClerk = () => {
      const win = window as unknown as { Clerk?: ClerkInstance }
      if (win.Clerk) {
        const clerk = win.Clerk
        setClerkState({
          isSignedIn: !!clerk.user,
          user: clerk.user || null,
          getToken: async () => {
            try {
              return clerk.session?.getToken() || null
            } catch {
              return null
            }
          },
        })
      }
    }

    // 初回チェック
    checkClerk()

    // Clerkがまだロードされていない場合は少し待って再試行
    const timer = setTimeout(checkClerk, 500)
    const timer2 = setTimeout(checkClerk, 1500)

    return () => {
      clearTimeout(timer)
      clearTimeout(timer2)
    }
  }, [])

  return clerkState
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

  // パーツブレンド状態
  const [partsBlendState, setPartsBlendState] = useState<PartsBlendState>({
    selection: defaultPartsSelection,
    image: null,
    isLoading: false,
    error: null,
  })

  // 表示モード（'morph' または 'parts'）
  const [viewMode, setViewMode] = useState<'morph' | 'parts'>('morph')

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

  /**
   * パーツ選択変更ハンドラ
   */
  const handlePartsSelectionChange = useCallback((selection: PartsSelection) => {
    setPartsBlendState((prev) => ({
      ...prev,
      selection,
      // 選択が変更されたら以前の結果をクリア
      image: null,
      error: null,
    }))
  }, [])

  /**
   * パーツブレンド実行ハンドラ
   */
  const handlePartsBlend = useCallback(async () => {
    const { currentImage, idealImage } = sourceImages

    if (!currentImage || !idealImage) {
      setPartsBlendState((prev) => ({
        ...prev,
        error: '画像データが見つかりません。',
      }))
      return
    }

    // 少なくとも1つのパーツが選択されているか確認
    const hasAnySelection = Object.values(partsBlendState.selection).some(Boolean)
    if (!hasAnySelection) {
      setPartsBlendState((prev) => ({
        ...prev,
        error: 'ブレンドするパーツを1つ以上選択してください。',
      }))
      return
    }

    setPartsBlendState((prev) => ({ ...prev, isLoading: true, error: null }))

    try {
      // Data URLをFileに変換
      const currentFile = dataUrlToFile(currentImage, 'current.png')
      const idealFile = dataUrlToFile(idealImage, 'ideal.png')

      // パーツブレンドAPIを呼び出し
      const result = await blendParts(currentFile, idealFile, partsBlendState.selection)

      setPartsBlendState((prev) => ({
        ...prev,
        image: toDataUrl(result.image),
        isLoading: false,
        error: null,
      }))
    } catch (error) {
      console.error('Parts blend error:', error)

      let errorMessage = 'パーツブレンド中にエラーが発生しました。'
      if (error instanceof ApiError) {
        errorMessage = error.localizedMessage
      } else if (error instanceof Error) {
        errorMessage = error.message
      }

      setPartsBlendState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }))
    }
  }, [sourceImages, partsBlendState.selection])

  /**
   * 選択されたパーツの表示名リストを取得
   */
  const selectedPartsNames = useMemo(() => {
    return (Object.entries(partsBlendState.selection) as [keyof PartsSelection, boolean][])
      .filter(([, isSelected]) => isSelected)
      .map(([part]) => PARTS_DISPLAY_NAMES[part])
  }, [partsBlendState.selection])

  return (
    <div className="min-h-screen flex flex-col bg-neutral-50">
      <Header />

      {/* メインコンテンツ */}
      <main className="flex-1 pt-20">
        <div className="container-narrow py-8 md:py-12">
          {/* ページタイトル */}
          <div className="text-center mb-8 md:mb-12">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">Result</p>
            <h1 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900">
              シミュレーション結果
            </h1>
          </div>

          {/* エラー表示 */}
          {state.error && (
            <div
              className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl"
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
                    className="mt-2 text-sm font-medium text-primary-600 hover:text-primary-800 underline"
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
              <div className="w-12 h-12 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin mb-4"></div>
              <p className="text-neutral-700 text-base font-serif">シミュレーション画像を生成中...</p>
              <p className="text-neutral-500 text-sm mt-2">しばらくお待ちください</p>
            </div>
          )}

          {/* 結果表示 */}
          {!state.isLoading && !state.error && state.images.length > 0 && (
            <>
              {/* 結果画像 */}
              <div
                className="bg-white rounded-2xl shadow-elegant p-4 mb-8"
                data-testid="result-image-container"
              >
                <div className="aspect-square max-w-md mx-auto overflow-hidden rounded-xl bg-neutral-100">
                  {viewMode === 'morph' ? (
                    // モーフィングモードの画像表示
                    currentImage ? (
                      <img
                        src={currentImage}
                        alt={`変化度${Math.round(state.currentProgress * 100)}%のシミュレーション結果`}
                        className="w-full h-full object-cover"
                        data-testid="result-image"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-neutral-400">
                        画像を読み込み中...
                      </div>
                    )
                  ) : (
                    // パーツブレンドモードの画像表示
                    partsBlendState.image ? (
                      <img
                        src={partsBlendState.image}
                        alt={`パーツブレンド結果（${selectedPartsNames.join('、')}）`}
                        className="w-full h-full object-cover"
                        data-testid="result-image"
                      />
                    ) : partsBlendState.isLoading ? (
                      <div className="w-full h-full flex flex-col items-center justify-center text-neutral-400">
                        <div className="w-10 h-10 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin mb-3"></div>
                        <p>パーツをブレンド中...</p>
                      </div>
                    ) : (
                      // パーツ未選択時は元の画像を表示
                      sourceImages.currentImage ? (
                        <img
                          src={sourceImages.currentImage}
                          alt="現在の顔"
                          className="w-full h-full object-cover opacity-70"
                          data-testid="result-image"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-neutral-400">
                          パーツを選択してブレンドを実行してください
                        </div>
                      )
                    )
                  )}
                </div>
              </div>

              {/* タブ切り替え */}
              <div className="flex justify-center mb-6">
                <div className="inline-flex rounded-full bg-neutral-100 p-1">
                  <button
                    type="button"
                    onClick={() => setViewMode('morph')}
                    className={`px-6 py-2 text-sm font-medium rounded-full transition-all duration-200 ${
                      viewMode === 'morph'
                        ? 'bg-white text-primary-700 shadow-sm'
                        : 'text-neutral-600 hover:text-neutral-900'
                    }`}
                    data-testid="view-mode-morph"
                  >
                    モーフィング
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode('parts')}
                    className={`px-6 py-2 text-sm font-medium rounded-full transition-all duration-200 ${
                      viewMode === 'parts'
                        ? 'bg-white text-primary-700 shadow-sm'
                        : 'text-neutral-600 hover:text-neutral-900'
                    }`}
                    data-testid="view-mode-parts"
                  >
                    パーツブレンド
                  </button>
                </div>
              </div>

              {/* モーフィングモード */}
              {viewMode === 'morph' && (
                <div className="mb-8">
                  <ResultSlider
                    value={state.currentProgress}
                    onChange={handleProgressChange}
                    disabled={state.isSaving || state.isSharing}
                    testId="result-slider"
                  />
                </div>
              )}

              {/* パーツブレンドモード */}
              {viewMode === 'parts' && (
                <div className="mb-8 space-y-6">
                  {/* パーツ選択UI */}
                  <div className="bg-white rounded-2xl shadow-sm p-6">
                    <PartsSelector
                      selection={partsBlendState.selection}
                      onChange={handlePartsSelectionChange}
                      disabled={partsBlendState.isLoading}
                      testId="parts-selector"
                    />

                    {/* パーツブレンド実行ボタン */}
                    <div className="mt-6 flex justify-center">
                      <button
                        type="button"
                        onClick={handlePartsBlend}
                        disabled={partsBlendState.isLoading || !Object.values(partsBlendState.selection).some(Boolean)}
                        className={`
                          px-8 py-3 text-base font-medium rounded-full
                          focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-300
                          ${
                            partsBlendState.isLoading
                              ? 'bg-neutral-300 text-neutral-500 cursor-wait'
                              : !Object.values(partsBlendState.selection).some(Boolean)
                                ? 'bg-neutral-200 text-neutral-400 cursor-not-allowed'
                                : 'bg-primary-700 text-white hover:bg-primary-800 hover:shadow-elegant focus:ring-primary-500'
                          }
                        `}
                        data-testid="parts-blend-button"
                      >
                        {partsBlendState.isLoading ? (
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
                            ブレンド中...
                          </span>
                        ) : (
                          'パーツをブレンド'
                        )}
                      </button>
                    </div>

                    {/* パーツブレンドエラー */}
                    {partsBlendState.error && (
                      <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                        {partsBlendState.error}
                      </div>
                    )}
                  </div>

                  {/* パーツブレンド結果画像 */}
                  {partsBlendState.image && (
                    <div className="bg-white rounded-2xl shadow-elegant p-4" data-testid="parts-blend-result">
                      <div className="text-center mb-3">
                        <p className="text-sm text-neutral-600">
                          適用パーツ: <span className="font-medium">{selectedPartsNames.join('、')}</span>
                        </p>
                      </div>
                      <div className="aspect-square max-w-md mx-auto overflow-hidden rounded-xl bg-neutral-100">
                        <img
                          src={partsBlendState.image}
                          alt={`パーツブレンド結果（${selectedPartsNames.join('、')}）`}
                          className="w-full h-full object-cover"
                          data-testid="parts-blend-image"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 保存・共有ボタン */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-6">
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={state.isSaving || !!state.savedSimulationId}
                  className={`
                    px-8 py-3 text-base font-medium rounded-full
                    focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-300
                    ${
                      state.savedSimulationId
                        ? 'bg-green-600 text-white cursor-default'
                        : state.isSaving
                          ? 'bg-neutral-300 text-neutral-500 cursor-wait'
                          : 'bg-primary-700 text-white hover:bg-primary-800 hover:shadow-elegant focus:ring-primary-500'
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
                    px-8 py-3 text-base font-medium rounded-full
                    focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-300
                    ${
                      state.isSharing
                        ? 'bg-neutral-300 text-neutral-500 cursor-wait'
                        : 'bg-white text-primary-700 border border-primary-300 hover:bg-primary-50 hover:border-primary-400 focus:ring-primary-500'
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
              <div className="flex justify-center mt-6">
                <button
                  type="button"
                  onClick={handleNewSimulation}
                  className="w-full max-w-md px-8 py-4 text-base font-medium text-neutral-600 bg-white border border-neutral-300 rounded-full hover:bg-neutral-50 hover:border-neutral-400 focus:outline-none focus:ring-2 focus:ring-neutral-400 focus:ring-offset-2 transition-all duration-300"
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
                className="mx-auto h-12 w-12 text-neutral-400 mb-4"
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
              <p className="text-neutral-600 mb-4">
                シミュレーション結果がありません
              </p>
              <button
                type="button"
                onClick={handleNewSimulation}
                className="px-6 py-3 text-sm font-medium text-white bg-primary-700 rounded-full hover:bg-primary-800 hover:shadow-elegant focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-all duration-300"
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

      <Footer />
    </div>
  )
}

/**
 * シミュレーション結果画面 (SCR-003)
 *
 * 参照: functional-spec.md セクション 3.4
 * 参照: business-spec.md UC-004, UC-005, UC-006, UC-007
 */
export default function SimulationResultPage() {
  const { isSignedIn, user, getToken } = useClerkState()

  return (
    <SimulationResultContent
      isSignedIn={isSignedIn}
      user={user}
      getToken={getToken}
    />
  )
}
