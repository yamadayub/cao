'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { LoginPromptModal } from '@/components/features/LoginPromptModal'
import { ShareUrlModal } from '@/components/features/ShareUrlModal'
import { PartsBlurOverlay, type LoginPromptInfo } from '@/components/features/PartsBlurOverlay'
import { ShareButton } from '@/components/features/ShareButton'
import { createSimulation, createShareUrl } from '@/lib/api/simulations'
import { swapAndWait, applySwapParts } from '@/lib/api/swap'
import { ApiError } from '@/lib/api/client'
import type { PartsSelection, SwapJobStatus, SwapPartsIntensity } from '@/lib/api/types'
import { PARTS_DISPLAY_NAMES } from '@/lib/api/types'
import { PartsSelector } from '@/components/features/PartsSelector'
import {
  savePendingAction,
  getPendingAction,
  clearPendingAction,
  getLoginPromptMessage,
  type PendingActionType,
} from '@/lib/pending-action'

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
  loadingProgress: number // ジョブ進捗 0-100
  loadingMessage: string  // 進捗メッセージ
  error: string | null
  isSaving: boolean
  isSharing: boolean
  savedSimulationId: string | null
  shareUrl: string | null
}

/**
 * パーツ適用状態の型
 */
interface PartsBlendState {
  selection: PartsSelection
  intensity: SwapPartsIntensity  // パーツごとの強度
  image: string | null
  isLoading: boolean
  error: string | null
}

/**
 * デフォルトのパーツ強度（全て1.0）
 */
const defaultPartsIntensity: SwapPartsIntensity = {
  left_eye: 1.0,
  right_eye: 1.0,
  left_eyebrow: 1.0,
  right_eyebrow: 1.0,
  nose: 1.0,
  lips: 1.0,
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
 * ログイン状態の変更を検知して自動更新
 */
function useClerkState() {
  const [clerkState, setClerkState] = useState<{
    isSignedIn: boolean
    wasSignedIn: boolean // 直前のログイン状態（ログイン完了検知用）
    justLoggedIn: boolean // ログインが完了したかどうか
    user: { primaryEmailAddress?: { emailAddress: string } | null } | null
    getToken: () => Promise<string | null>
  }>({
    isSignedIn: false,
    wasSignedIn: false,
    justLoggedIn: false,
    user: null,
    getToken: async () => null,
  })

  useEffect(() => {
    const updateClerkState = () => {
      const win = window as unknown as { Clerk?: ClerkInstance }
      if (win.Clerk) {
        const clerk = win.Clerk
        setClerkState(prev => ({
          isSignedIn: !!clerk.user,
          wasSignedIn: prev.isSignedIn,
          justLoggedIn: false,
          user: clerk.user || null,
          getToken: async () => {
            try {
              return clerk.session?.getToken() || null
            } catch {
              return null
            }
          },
        }))
      }
    }

    // 初回チェック
    updateClerkState()

    // Clerkがまだロードされていない場合は少し待って再試行
    const timer = setTimeout(updateClerkState, 500)
    const timer2 = setTimeout(updateClerkState, 1500)

    // Clerkの認証状態変更を監視（ポーリング）
    // Clerkのモーダルログイン後に状態を更新するため
    const pollInterval = setInterval(() => {
      const win = window as unknown as { Clerk?: ClerkInstance }
      if (win.Clerk) {
        const currentSignedIn = !!win.Clerk.user
        setClerkState(prev => {
          // ログイン状態が変化した場合のみ更新
          if (prev.isSignedIn !== currentSignedIn) {
            const justLoggedIn = !prev.isSignedIn && currentSignedIn
            return {
              isSignedIn: currentSignedIn,
              wasSignedIn: prev.isSignedIn,
              justLoggedIn,
              user: win.Clerk?.user || null,
              getToken: async () => {
                try {
                  return win.Clerk?.session?.getToken() || null
                } catch {
                  return null
                }
              },
            }
          }
          return prev
        })
      }
    }, 1000)

    return () => {
      clearTimeout(timer)
      clearTimeout(timer2)
      clearInterval(pollInterval)
    }
  }, [])

  // justLoggedInフラグをリセット
  const resetJustLoggedIn = useCallback(() => {
    setClerkState(prev => ({ ...prev, justLoggedIn: false }))
  }, [])

  return { ...clerkState, resetJustLoggedIn }
}

/**
 * シミュレーション結果画面の本体
 */
interface SimulationResultContentProps {
  isSignedIn: boolean
  justLoggedIn: boolean
  resetJustLoggedIn: () => void
  user: { primaryEmailAddress?: { emailAddress: string } | null } | null
  getToken: () => Promise<string | null>
}

function SimulationResultContent({ isSignedIn, justLoggedIn, resetJustLoggedIn, user, getToken }: SimulationResultContentProps) {
  const router = useRouter()

  // 状態管理
  const [state, setState] = useState<ResultState>({
    images: [],
    currentProgress: 0.5,
    isLoading: true,
    loadingProgress: 0,
    loadingMessage: '',
    error: null,
    isSaving: false,
    isSharing: false,
    savedSimulationId: null,
    shareUrl: null,
  })

  // モーダル状態
  const [showLoginModal, setShowLoginModal] = useState(false)
  const [showShareModal, setShowShareModal] = useState(false)
  const [loginAction, setLoginAction] = useState<'save' | 'share' | 'parts-blur' | null>(null)
  const [partsLoginPromptInfo, setPartsLoginPromptInfo] = useState<LoginPromptInfo | null>(null)

  // 元画像のData URL
  const [sourceImages, setSourceImages] = useState<{
    currentImage: string | null
    idealImage: string | null
  }>({ currentImage: null, idealImage: null })

  // パーツ適用状態
  const [partsBlendState, setPartsBlendState] = useState<PartsBlendState>({
    selection: defaultPartsSelection,
    intensity: defaultPartsIntensity,
    image: null,
    isLoading: false,
    error: null,
  })

  // Face Swapの結果画像（パーツ合成のベース）
  const [swappedImage, setSwappedImage] = useState<string | null>(null)

  // API呼び出し中かどうかのref（重複リクエスト防止）
  const isGeneratingRef = useRef(false)

  // 表示モード（'morph' または 'parts'）
  const [viewMode, setViewMode] = useState<'morph' | 'parts'>('morph')

  // パーツモードの表示切替（'current' または 'applied'）
  const [partsViewMode, setPartsViewMode] = useState<'current' | 'applied'>('applied')

  /**
   * 現在の変化度に対応する画像を取得
   */
  const currentImage = useMemo(() => {
    const image = state.images.find((img) => img.progress === state.currentProgress)
    return image?.image || null
  }, [state.images, state.currentProgress])

  /**
   * Face Swap画像を生成（Replicate API使用）
   */
  const generateMorphImages = useCallback(async () => {
    // 重複リクエスト防止
    if (isGeneratingRef.current) {
      console.log('Already generating, skipping duplicate request')
      return
    }
    isGeneratingRef.current = true

    const { currentImage, idealImage } = getStoredImages()

    if (!currentImage || !idealImage) {
      isGeneratingRef.current = false
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: '画像データが見つかりません。シミュレーション作成画面からやり直してください。',
      }))
      return
    }

    setSourceImages({ currentImage, idealImage })

    setState((prev) => ({
      ...prev,
      isLoading: true,
      loadingProgress: 0,
      loadingMessage: '処理中...',
      error: null
    }))

    try {
      // Base64データを取得（Data URLプレフィックスを除去）
      const base64Current = currentImage.startsWith('data:')
        ? currentImage.split(',')[1]
        : currentImage
      const base64Ideal = idealImage.startsWith('data:')
        ? idealImage.split(',')[1]
        : idealImage

      // Face Swap APIを呼び出し
      setState((prev) => ({
        ...prev,
        loadingProgress: 10,
        loadingMessage: '処理中...',
      }))

      console.log('Starting Face Swap API call...')
      console.log('current_image length:', base64Current.length)
      console.log('ideal_image length:', base64Ideal.length)

      const swapResult = await swapAndWait({
        current_image: base64Current,
        ideal_image: base64Ideal,
      }, {
        onProgress: (status: SwapJobStatus) => {
          console.log('Swap progress:', status)
          let progress = 10
          if (status === 'processing') progress = 50
          if (status === 'completed') progress = 90
          setState((prev) => ({
            ...prev,
            loadingProgress: progress,
          }))
        },
      })

      console.log('Face Swap API response received')
      console.log('swapResult.status:', swapResult.status)
      console.log('swapResult.swapped_image exists:', !!swapResult.swapped_image)
      console.log('swapResult.swapped_image length:', swapResult.swapped_image?.length)

      if (!swapResult.swapped_image) {
        throw new Error('Face Swap結果が取得できませんでした')
      }

      // スワップ結果を保存（JPEG形式で返却される）
      const swappedDataUrl = swapResult.swapped_image.startsWith('data:')
        ? swapResult.swapped_image
        : `data:image/jpeg;base64,${swapResult.swapped_image}`

      // 画像が実際に読み込めるかを検証
      console.log('Validating image data URL...')
      await new Promise<void>((resolve, reject) => {
        const img = new Image()
        img.onload = () => {
          console.log('Image validated successfully:', img.width, 'x', img.height)
          resolve()
        }
        img.onerror = (e) => {
          console.error('Image validation failed:', e)
          reject(new Error('画像データの読み込みに失敗しました。もう一度お試しください。'))
        }
        img.src = swappedDataUrl
      })

      setSwappedImage(swappedDataUrl)

      // 結果画像を生成（現在と理想の2段階）
      const generatedImages: ImageData[] = [
        {
          progress: 0,
          image: currentImage, // 現在
        },
        {
          progress: 1.0,
          image: swappedDataUrl, // 理想
        },
      ]

      setState((prev) => ({
        ...prev,
        images: generatedImages,
        currentProgress: 1.0, // デフォルトで100%を表示
        isLoading: false,
        loadingProgress: 100,
        loadingMessage: '',
        error: null,
      }))
    } catch (error) {
      console.error('Face Swap error:', error)
      console.error('Error type:', error?.constructor?.name)
      console.error('Error details:', {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        code: error instanceof ApiError ? error.code : undefined,
      })

      let errorMessage = '画像の生成中にエラーが発生しました。'
      if (error instanceof ApiError) {
        errorMessage = error.localizedMessage
        console.error('ApiError code:', error.code)
        console.error('ApiError statusCode:', error.statusCode)
      } else if (error instanceof Error) {
        errorMessage = error.message
      }

      setState((prev) => ({
        ...prev,
        isLoading: false,
        loadingProgress: 0,
        loadingMessage: '',
        error: errorMessage,
      }))
    } finally {
      isGeneratingRef.current = false
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
      savePendingAction({
        type: 'save',
        viewMode,
        partsSelection: partsBlendState.selection,
      })
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
      savePendingAction({
        type: 'share',
        viewMode,
        partsSelection: partsBlendState.selection,
      })
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
   * SNSシェア用の画像を取得
   */
  const getShareImage = useCallback((): string => {
    if (viewMode === 'morph') {
      // モーフィングモードでは現在表示中の画像
      return currentImage || ''
    } else {
      // パーツモードでは適用後の画像
      return partsBlendState.image || sourceImages.currentImage || ''
    }
  }, [viewMode, currentImage, partsBlendState.image, sourceImages.currentImage])

  /**
   * 適用パーツのリストを取得
   */
  const getAppliedParts = useCallback((): string[] => {
    if (viewMode === 'parts') {
      return Object.entries(partsBlendState.selection)
        .filter(([, selected]) => selected)
        .map(([part]) => part)
    }
    return []
  }, [viewMode, partsBlendState.selection])

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
   * パーツ別ブラー画像クリックハンドラ
   */
  const handlePartsBlurLoginClick = useCallback((info?: LoginPromptInfo) => {
    setLoginAction('parts-blur')
    savePendingAction({
      type: 'parts-blur',
      viewMode: 'parts',
      partsSelection: partsBlendState.selection,
    })
    if (info) {
      setPartsLoginPromptInfo(info)
    }
    setShowLoginModal(true)
  }, [partsBlendState.selection])

  /**
   * 再試行ハンドラ
   */
  const handleRetry = useCallback(() => {
    generateMorphImages()
  }, [generateMorphImages])

  /**
   * 画像ダウンロードハンドラ
   */
  const handleDownload = useCallback(() => {
    let imageToDownload: string | null = null
    let filename = 'cao-simulation'

    if (viewMode === 'morph') {
      // モーフィングモードでは現在表示中の画像をダウンロード
      imageToDownload = currentImage
      filename = state.currentProgress === 0 ? 'cao-current' : 'cao-ideal'
    } else {
      // パーツモードでは適用後の画像または現在の画像をダウンロード
      if (partsViewMode === 'applied' && partsBlendState.image) {
        imageToDownload = partsBlendState.image
        filename = 'cao-parts-applied'
      } else {
        imageToDownload = sourceImages.currentImage
        filename = 'cao-current'
      }
    }

    if (!imageToDownload) return

    // Data URLからダウンロード
    const link = document.createElement('a')
    link.href = imageToDownload
    link.download = `${filename}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [viewMode, currentImage, state.currentProgress, partsViewMode, partsBlendState.image, sourceImages.currentImage])

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
   * パーツ適用実行ハンドラ（Face Swap Parts API使用）
   */
  const handlePartsBlend = useCallback(async () => {
    const { currentImage } = sourceImages

    if (!currentImage || !swappedImage) {
      setPartsBlendState((prev) => ({
        ...prev,
        error: swappedImage ? '画像データが見つかりません。' : 'Face Swapを先に実行してください。',
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
      // Base64データを取得（Data URLプレフィックスを除去）
      const base64Current = currentImage.startsWith('data:')
        ? currentImage.split(',')[1]
        : currentImage
      const base64Swapped = swappedImage.startsWith('data:')
        ? swappedImage.split(',')[1]
        : swappedImage

      // 選択されたパーツの強度を設定（選択されていないパーツは0）
      const partsIntensity: SwapPartsIntensity = {
        left_eye: partsBlendState.selection.left_eye ? partsBlendState.intensity.left_eye : 0,
        right_eye: partsBlendState.selection.right_eye ? partsBlendState.intensity.right_eye : 0,
        left_eyebrow: partsBlendState.selection.left_eyebrow ? partsBlendState.intensity.left_eyebrow : 0,
        right_eyebrow: partsBlendState.selection.right_eyebrow ? partsBlendState.intensity.right_eyebrow : 0,
        nose: partsBlendState.selection.nose ? partsBlendState.intensity.nose : 0,
        lips: partsBlendState.selection.lips ? partsBlendState.intensity.lips : 0,
      }

      // Face Swap Parts APIを使用
      const result = await applySwapParts({
        current_image: base64Current,
        swapped_image: base64Swapped,
        parts: partsIntensity,
      })

      if (result.result_image) {
        setPartsBlendState((prev) => ({
          ...prev,
          image: result.result_image.startsWith('data:')
            ? result.result_image
            : `data:image/png;base64,${result.result_image}`,
          isLoading: false,
          error: null,
        }))
      }
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
  }, [sourceImages, swappedImage, partsBlendState.selection, partsBlendState.intensity])

  /**
   * ログイン完了時に保留アクションを実行
   */
  useEffect(() => {
    if (!justLoggedIn) return

    const pendingAction = getPendingAction()
    if (!pendingAction) {
      resetJustLoggedIn()
      return
    }

    // 保留アクションをクリア
    clearPendingAction()
    resetJustLoggedIn()

    // 表示モードを復元
    if (pendingAction.viewMode) {
      setViewMode(pendingAction.viewMode)
    }

    // パーツ選択状態を復元
    if (pendingAction.partsSelection) {
      setPartsBlendState(prev => ({
        ...prev,
        selection: pendingAction.partsSelection!,
      }))
    }

    // アクションを実行
    switch (pendingAction.type) {
      case 'parts-blur':
        // パーツモードに切り替え、ブラーは認証済みなので自動解除
        setViewMode('parts')
        setPartsViewMode('applied')
        break
      case 'download':
        // ダウンロードを実行（少し遅延させてUIの更新を待つ）
        setTimeout(() => {
          handleDownload()
        }, 100)
        break
      case 'save':
        // 保存を実行
        handleSave()
        break
      case 'share':
        // 共有を実行
        handleShare()
        break
    }
  }, [justLoggedIn, resetJustLoggedIn, handleDownload, handleSave, handleShare])

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

          {/* ローディング表示（進捗付き） */}
          {state.isLoading && (
            <div
              className="flex flex-col items-center justify-center py-16"
              data-testid="loading"
            >
              <div className="w-12 h-12 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin mb-4"></div>
              <p className="text-neutral-700 text-base font-serif">
                {state.loadingMessage || 'シミュレーション画像を生成中...'}
              </p>
              {/* 進捗バー */}
              <div className="w-64 mt-4">
                <div className="bg-neutral-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${state.loadingProgress}%` }}
                  />
                </div>
                <p className="text-neutral-500 text-sm mt-2 text-center">
                  {state.loadingProgress}%
                </p>
              </div>
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
                    partsBlendState.isLoading ? (
                      <div className="w-full h-full flex flex-col items-center justify-center text-neutral-400">
                        <div className="w-10 h-10 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin mb-3"></div>
                        <p>処理中...</p>
                      </div>
                    ) : partsViewMode === 'current' ? (
                      // 現在の画像を表示
                      sourceImages.currentImage ? (
                        <img
                          src={sourceImages.currentImage}
                          alt="現在の顔"
                          className="w-full h-full object-cover"
                          data-testid="result-image"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-neutral-400">
                          画像を読み込み中...
                        </div>
                      )
                    ) : (
                      // 適用後の画像を表示
                      partsBlendState.image ? (
                        // パーツ適用結果がある場合
                        <PartsBlurOverlay
                          imageUrl={partsBlendState.image}
                          altText={`パーツ適用結果（${selectedPartsNames.join('、')}）`}
                          isAuthenticated={isSignedIn}
                          onLoginClick={handlePartsBlurLoginClick}
                          isLoading={false}
                          showPartsLabel={true}
                          appliedParts={Object.entries(partsBlendState.selection)
                            .filter(([, selected]) => selected)
                            .map(([part]) => part)}
                          testId="parts-blur-overlay"
                        />
                      ) : sourceImages.currentImage ? (
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
                    全体
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
                    パーツ別適用
                  </button>
                </div>
              </div>

              {/* 全体モード */}
              {viewMode === 'morph' && (
                <div className="mb-8">
                  <div className="flex justify-center gap-4">
                    <button
                      type="button"
                      onClick={() => handleProgressChange(0)}
                      disabled={state.isSaving || state.isSharing}
                      className={`px-8 py-3 text-base font-medium rounded-full transition-all duration-200 ${
                        state.currentProgress === 0
                          ? 'bg-primary-700 text-white shadow-md'
                          : 'bg-white text-neutral-600 border border-neutral-300 hover:bg-neutral-50'
                      } ${(state.isSaving || state.isSharing) ? 'opacity-50 cursor-not-allowed' : ''}`}
                      data-testid="view-current"
                    >
                      現在
                    </button>
                    <button
                      type="button"
                      onClick={() => handleProgressChange(1.0)}
                      disabled={state.isSaving || state.isSharing}
                      className={`px-8 py-3 text-base font-medium rounded-full transition-all duration-200 ${
                        state.currentProgress === 1.0
                          ? 'bg-primary-700 text-white shadow-md'
                          : 'bg-white text-neutral-600 border border-neutral-300 hover:bg-neutral-50'
                      } ${(state.isSaving || state.isSharing) ? 'opacity-50 cursor-not-allowed' : ''}`}
                      data-testid="view-ideal"
                    >
                      理想
                    </button>
                  </div>
                </div>
              )}

              {/* パーツ別適用モード */}
              {viewMode === 'parts' && (
                <div className="mb-8 space-y-6">
                  {/* 現在/適用後の切り替えボタン */}
                  <div className="flex justify-center gap-4">
                    <button
                      type="button"
                      onClick={() => setPartsViewMode('current')}
                      disabled={partsBlendState.isLoading}
                      className={`px-8 py-3 text-base font-medium rounded-full transition-all duration-200 ${
                        partsViewMode === 'current'
                          ? 'bg-primary-700 text-white shadow-md'
                          : 'bg-white text-neutral-600 border border-neutral-300 hover:bg-neutral-50'
                      } ${partsBlendState.isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      data-testid="parts-view-current"
                    >
                      現在
                    </button>
                    <button
                      type="button"
                      onClick={() => setPartsViewMode('applied')}
                      disabled={partsBlendState.isLoading}
                      className={`px-8 py-3 text-base font-medium rounded-full transition-all duration-200 ${
                        partsViewMode === 'applied'
                          ? 'bg-primary-700 text-white shadow-md'
                          : 'bg-white text-neutral-600 border border-neutral-300 hover:bg-neutral-50'
                      } ${partsBlendState.isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                      data-testid="parts-view-applied"
                    >
                      適用後
                    </button>
                  </div>

                  {/* 適用パーツ表示 */}
                  {partsBlendState.image && partsViewMode === 'applied' && (
                    <div className="text-center">
                      <p className="text-sm text-neutral-600">
                        適用パーツ: <span className="font-medium">{selectedPartsNames.join('、')}</span>
                      </p>
                    </div>
                  )}

                  {/* パーツ選択UI */}
                  <div className="bg-white rounded-2xl shadow-sm p-6">
                    <PartsSelector
                      selection={partsBlendState.selection}
                      onChange={handlePartsSelectionChange}
                      disabled={partsBlendState.isLoading}
                      testId="parts-selector"
                    />

                    {/* パーツ適用実行ボタン */}
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
                            処理中...
                          </span>
                        ) : (
                          '適用'
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
                </div>
              )}

              {/* ダウンロード・シェアボタン */}
              <div className="flex flex-col gap-4 max-w-md mx-auto mb-6">
                {/* シェアボタン（Web Share API / クリップボードコピー） */}
                <ShareButton
                  beforeImage={sourceImages.currentImage || ''}
                  afterImage={getShareImage()}
                  appliedParts={getAppliedParts()}
                  testId="share-button"
                />

                {/* ダウンロードボタン */}
                <button
                  type="button"
                  onClick={handleDownload}
                  disabled={state.isSharing}
                  className={`
                    w-full px-8 py-3 text-base font-medium rounded-xl
                    focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-300
                    ${
                      state.isSharing
                        ? 'bg-neutral-300 text-neutral-500 cursor-not-allowed'
                        : 'bg-white text-neutral-700 border border-neutral-300 hover:bg-neutral-50 hover:border-neutral-400 focus:ring-neutral-400'
                    }
                  `}
                  data-testid="download-button"
                >
                  <span className="flex items-center justify-center gap-2">
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
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                      />
                    </svg>
                    ダウンロード
                  </span>
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
        onClose={() => {
          setShowLoginModal(false)
          setPartsLoginPromptInfo(null)
          // モーダルを閉じた場合は保留アクションをクリア
          clearPendingAction()
        }}
        onLogin={handleLogin}
        title={
          loginAction === 'parts-blur' && partsLoginPromptInfo
            ? partsLoginPromptInfo.title
            : loginAction
              ? getLoginPromptMessage(loginAction).title
              : 'ログインが必要です'
        }
        description={
          loginAction === 'parts-blur' && partsLoginPromptInfo
            ? partsLoginPromptInfo.description
            : loginAction
              ? getLoginPromptMessage(loginAction).description
              : 'この機能を使用するにはログインが必要です。'
        }
        testId={loginAction === 'parts-blur' ? 'parts-login-prompt-modal' : 'login-prompt-modal'}
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
  const { isSignedIn, justLoggedIn, resetJustLoggedIn, user, getToken } = useClerkState()

  return (
    <SimulationResultContent
      isSignedIn={isSignedIn}
      justLoggedIn={justLoggedIn}
      resetJustLoggedIn={resetJustLoggedIn}
      user={user}
      getToken={getToken}
    />
  )
}
