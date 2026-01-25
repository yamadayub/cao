'use client'

import { useCallback, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { ImageUploader } from '@/components/features/ImageUploader'
import { CameraCapture } from '@/components/features/CameraCapture'
import { TermsAgreementModal } from '@/components/features/TermsAgreementModal'
import { useTermsAgreement } from '@/hooks/useTermsAgreement'

/**
 * 画像のステート型
 */
interface ImageState {
  file: File | null
  previewUrl: string | null
}

/**
 * 現在のステップ
 */
type Step = 'upload-ideal' | 'select-current-method' | 'upload-current' | 'review'

/**
 * シミュレーション作成画面 (SCR-002)
 *
 * フロー:
 * 1. 理想の顔をアップロード
 * 2. 現在の顔の取得方法を選択（カメラ or アップロード）
 * 3. 確認して生成
 */
export default function SimulatePage() {
  const router = useRouter()
  const { hasAgreed, isLoading: isLoadingTerms, agree } = useTermsAgreement()

  // 現在のステップ
  const [step, setStep] = useState<Step>('upload-ideal')

  // 理想の顔画像の状態
  const [idealImage, setIdealImage] = useState<ImageState>({
    file: null,
    previewUrl: null,
  })

  // 現在の顔画像の状態
  const [currentImage, setCurrentImage] = useState<ImageState>({
    file: null,
    previewUrl: null,
  })

  // エラー状態
  const [idealError, setIdealError] = useState<string | undefined>()
  const [currentError, setCurrentError] = useState<string | undefined>()

  // 生成中フラグ
  const [isGenerating, setIsGenerating] = useState(false)

  // 利用規約同意モーダル表示状態
  const [isTermsModalOpen, setIsTermsModalOpen] = useState(false)

  // カメラモード表示状態
  const [showCamera, setShowCamera] = useState(false)

  /**
   * 画像アップロードを試みる（未同意の場合はモーダル表示）
   */
  const attemptUpload = useCallback(() => {
    if (!hasAgreed) {
      setIsTermsModalOpen(true)
      return false
    }
    return true
  }, [hasAgreed])

  /**
   * 利用規約同意モーダルを閉じる
   */
  const handleTermsModalClose = useCallback(() => {
    setIsTermsModalOpen(false)
  }, [])

  /**
   * 利用規約に同意する
   */
  const handleTermsAgree = useCallback(() => {
    agree()
    setIsTermsModalOpen(false)
  }, [agree])

  /**
   * 理想の顔画像選択ハンドラ
   */
  const handleIdealFileSelect = useCallback(
    (file: File, previewUrl: string) => {
      setIdealImage({ file, previewUrl })
      setIdealError(undefined)
    },
    []
  )

  /**
   * 理想の顔画像削除ハンドラ
   */
  const handleIdealFileRemove = useCallback(() => {
    setIdealImage({ file: null, previewUrl: null })
    setIdealError(undefined)
    // 現在の顔もリセット
    setCurrentImage({ file: null, previewUrl: null })
    setStep('upload-ideal')
  }, [])

  /**
   * 理想の顔画像バリデーションエラーハンドラ
   */
  const handleIdealValidationError = useCallback((error: string) => {
    setIdealError(error)
  }, [])

  /**
   * 次のステップへ（方法選択）
   */
  const handleProceedToSelectMethod = useCallback(() => {
    if (!idealImage.previewUrl) return
    setStep('select-current-method')
  }, [idealImage.previewUrl])

  /**
   * カメラモードを選択
   */
  const handleSelectCamera = useCallback(() => {
    setShowCamera(true)
  }, [])

  /**
   * アップロードモードを選択
   */
  const handleSelectUpload = useCallback(() => {
    setStep('upload-current')
  }, [])

  /**
   * カメラで撮影完了
   */
  const handleCameraCapture = useCallback((file: File, previewUrl: string) => {
    setCurrentImage({ file, previewUrl })
    setShowCamera(false)
    setStep('review')
  }, [])

  /**
   * カメラをキャンセル
   */
  const handleCameraCancel = useCallback(() => {
    setShowCamera(false)
  }, [])

  /**
   * 現在の顔画像選択ハンドラ（アップロード）
   */
  const handleCurrentFileSelect = useCallback(
    (file: File, previewUrl: string) => {
      setCurrentImage({ file, previewUrl })
      setCurrentError(undefined)
      setStep('review')
    },
    []
  )

  /**
   * 現在の顔画像削除ハンドラ
   */
  const handleCurrentFileRemove = useCallback(() => {
    setCurrentImage({ file: null, previewUrl: null })
    setCurrentError(undefined)
  }, [])

  /**
   * 現在の顔画像バリデーションエラーハンドラ
   */
  const handleCurrentValidationError = useCallback((error: string) => {
    setCurrentError(error)
  }, [])

  /**
   * 現在の顔を変更する（方法選択に戻る）
   */
  const handleChangeCurrentImage = useCallback(() => {
    setCurrentImage({ file: null, previewUrl: null })
    setStep('select-current-method')
  }, [])

  /**
   * 理想の顔を変更する
   */
  const handleChangeIdealImage = useCallback(() => {
    setIdealImage({ file: null, previewUrl: null })
    setCurrentImage({ file: null, previewUrl: null })
    setStep('upload-ideal')
  }, [])

  /**
   * 方法選択に戻る
   */
  const handleBackToSelectMethod = useCallback(() => {
    setStep('select-current-method')
  }, [])

  /**
   * シミュレーション生成ハンドラ
   */
  const handleGenerate = useCallback(async () => {
    if (!currentImage.file || !idealImage.file) {
      return
    }

    setIsGenerating(true)

    try {
      // TODO: 実際のAPI呼び出しを実装
      await new Promise((resolve) => setTimeout(resolve, 500))

      // 画像データをsessionStorageに保存（結果画面で使用）
      if (currentImage.previewUrl && idealImage.previewUrl) {
        sessionStorage.setItem('cao_current_image', currentImage.previewUrl)
        sessionStorage.setItem('cao_ideal_image', idealImage.previewUrl)
      }

      router.push('/simulate/result')
    } catch (error) {
      console.error('シミュレーション生成エラー:', error)
      // TODO: エラー処理を実装
    } finally {
      setIsGenerating(false)
    }
  }, [currentImage, idealImage, router])

  // ローディング中は簡易表示
  if (isLoadingTerms) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="w-8 h-8 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin"></div>
      </div>
    )
  }

  // カメラモード
  if (showCamera && idealImage.previewUrl) {
    return (
      <CameraCapture
        guideImageUrl={idealImage.previewUrl}
        onCapture={handleCameraCapture}
        onCancel={handleCameraCancel}
        testId="camera-capture"
      />
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-neutral-50">
      <Header />

      {/* メインコンテンツ */}
      <main className="flex-1 pt-20">
        <div className="container-narrow py-8 md:py-12">
          {/* ページタイトル */}
          <div className="text-center mb-8 md:mb-12">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">Create Simulation</p>
            <h1 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900 mb-3">
              シミュレーション作成
            </h1>
            <p className="text-sm md:text-base text-neutral-500">
              {step === 'upload-ideal' && '理想の顔写真をアップロードしてください'}
              {step === 'select-current-method' && '現在の顔の取得方法を選択してください'}
              {step === 'upload-current' && '現在の顔写真をアップロードしてください'}
              {step === 'review' && '写真を確認してください'}
            </p>
          </div>

          {/* 利用規約同意バナー（未同意の場合） */}
          {!hasAgreed && (
            <div
              className="mb-6 p-4 bg-primary-50 border border-primary-200 rounded-xl"
              data-testid="terms-agreement-banner"
            >
              <div className="flex items-start gap-3">
                <svg
                  className="w-5 h-5 text-primary-600 mt-0.5 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div>
                  <p className="text-sm text-primary-800">
                    画像をアップロードするには、
                    <button
                      type="button"
                      onClick={() => setIsTermsModalOpen(true)}
                      className="font-medium underline hover:text-primary-900"
                    >
                      利用規約への同意
                    </button>
                    が必要です。
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* ステップインジケーター */}
          <div className="flex items-center justify-center gap-4 mb-8">
            <div className={`flex items-center gap-2 ${step === 'upload-ideal' ? 'text-primary-700' : 'text-neutral-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                step === 'upload-ideal' ? 'bg-primary-700 text-white' : idealImage.file ? 'bg-primary-200 text-primary-700' : 'bg-neutral-200 text-neutral-500'
              }`}>
                {idealImage.file ? '✓' : '1'}
              </div>
              <span className="text-sm hidden sm:inline">理想の顔</span>
            </div>
            <div className="w-8 h-px bg-neutral-300" />
            <div className={`flex items-center gap-2 ${step === 'select-current-method' || step === 'upload-current' || step === 'review' ? 'text-primary-700' : 'text-neutral-400'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                step === 'select-current-method' || step === 'upload-current' ? 'bg-primary-700 text-white' : currentImage.file ? 'bg-primary-200 text-primary-700' : 'bg-neutral-200 text-neutral-500'
              }`}>
                {currentImage.file ? '✓' : '2'}
              </div>
              <span className="text-sm hidden sm:inline">現在の顔</span>
            </div>
          </div>

          {/* Step 1: 理想の顔をアップロード */}
          {step === 'upload-ideal' && (
            <div className="flex flex-col items-center gap-6">
              <div className="w-full max-w-[320px] bg-white rounded-2xl shadow-elegant p-6">
                <ImageUploader
                  label="理想の顔"
                  previewUrl={idealImage.previewUrl}
                  error={idealError}
                  onFileSelect={handleIdealFileSelect}
                  onFileRemove={handleIdealFileRemove}
                  onValidationError={handleIdealValidationError}
                  onClickBeforeSelect={attemptUpload}
                  disabled={isGenerating}
                  testId="ideal-image"
                />
              </div>

              {/* 次へボタン */}
              {idealImage.file && (
                <button
                  type="button"
                  onClick={handleProceedToSelectMethod}
                  className="px-8 py-4 bg-primary-700 text-white rounded-full font-medium hover:bg-primary-800 hover:shadow-elegant transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                  data-testid="proceed-button"
                >
                  次へ進む
                </button>
              )}

              {/* 説明テキスト */}
              <div className="text-center text-sm text-neutral-500 max-w-md">
                <p>
                  理想の顔写真をアップロードしてください。
                </p>
              </div>
            </div>
          )}

          {/* Step 2: 現在の顔の取得方法を選択 */}
          {step === 'select-current-method' && (
            <div className="flex flex-col items-center gap-6">
              {/* 理想の顔プレビュー */}
              <div className="w-full max-w-[200px] bg-white rounded-xl shadow-elegant p-3">
                <p className="text-xs text-neutral-500 text-center mb-2">理想の顔</p>
                <div className="aspect-square rounded-lg overflow-hidden">
                  {idealImage.previewUrl && (
                    <img
                      src={idealImage.previewUrl}
                      alt="理想の顔"
                      className="w-full h-full object-cover"
                    />
                  )}
                </div>
              </div>

              {/* 選択肢 */}
              <div className="w-full max-w-md space-y-4">
                <p className="text-center text-neutral-700 font-medium mb-4">
                  現在の顔をどのように取得しますか？
                </p>

                {/* カメラで撮影 */}
                <button
                  type="button"
                  onClick={handleSelectCamera}
                  className="w-full p-6 bg-white rounded-2xl shadow-elegant hover:shadow-lg transition-all duration-300 text-left group"
                  data-testid="select-camera-button"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-primary-100 rounded-xl flex items-center justify-center group-hover:bg-primary-200 transition-colors">
                      <svg className="w-7 h-7 text-primary-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-neutral-900 mb-1">カメラで撮影</h3>
                      <p className="text-sm text-neutral-500">
                        理想の顔をガイドにして撮影できます
                      </p>
                    </div>
                    <svg className="w-5 h-5 text-neutral-400 group-hover:text-primary-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </button>

                {/* 画像をアップロード */}
                <button
                  type="button"
                  onClick={handleSelectUpload}
                  className="w-full p-6 bg-white rounded-2xl shadow-elegant hover:shadow-lg transition-all duration-300 text-left group"
                  data-testid="select-upload-button"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-neutral-100 rounded-xl flex items-center justify-center group-hover:bg-neutral-200 transition-colors">
                      <svg className="w-7 h-7 text-neutral-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-neutral-900 mb-1">画像をアップロード</h3>
                      <p className="text-sm text-neutral-500">
                        手持ちの写真を使用します
                      </p>
                    </div>
                    <svg className="w-5 h-5 text-neutral-400 group-hover:text-primary-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </button>
              </div>

              {/* 戻るリンク */}
              <button
                type="button"
                onClick={handleChangeIdealImage}
                className="text-sm text-neutral-500 hover:text-primary-600 transition-colors"
              >
                理想の顔を変更する
              </button>
            </div>
          )}

          {/* Step 2b: 現在の顔をアップロード */}
          {step === 'upload-current' && (
            <div className="flex flex-col items-center gap-6">
              {/* 理想の顔プレビュー（小さく表示） */}
              <div className="w-full max-w-[160px] bg-white rounded-xl shadow-elegant p-2">
                <p className="text-xs text-neutral-500 text-center mb-1">理想の顔</p>
                <div className="aspect-square rounded-lg overflow-hidden">
                  {idealImage.previewUrl && (
                    <img
                      src={idealImage.previewUrl}
                      alt="理想の顔"
                      className="w-full h-full object-cover"
                    />
                  )}
                </div>
              </div>

              <div className="w-full max-w-[320px] bg-white rounded-2xl shadow-elegant p-6">
                <ImageUploader
                  label="現在の顔"
                  previewUrl={currentImage.previewUrl}
                  error={currentError}
                  onFileSelect={handleCurrentFileSelect}
                  onFileRemove={handleCurrentFileRemove}
                  onValidationError={handleCurrentValidationError}
                  onClickBeforeSelect={attemptUpload}
                  disabled={isGenerating}
                  testId="current-image"
                />
              </div>

              {/* 説明テキスト */}
              <div className="text-center text-sm text-neutral-500 max-w-md">
                <p>
                  理想の顔と同じ向き・角度の写真をお選びください。
                </p>
              </div>

              {/* 戻るリンク */}
              <button
                type="button"
                onClick={handleBackToSelectMethod}
                className="text-sm text-neutral-500 hover:text-primary-600 transition-colors"
              >
                戻る
              </button>
            </div>
          )}

          {/* Step 3: 確認画面 */}
          {step === 'review' && (
            <div className="flex flex-col items-center gap-6">
              {/* 2つの画像を並べて表示 */}
              <div className="flex flex-col md:flex-row gap-6 justify-center items-center">
                {/* 理想の顔 */}
                <div className="w-full max-w-[280px]">
                  <div className="bg-white rounded-2xl shadow-elegant p-4">
                    <h3 className="font-serif text-lg text-neutral-800 text-center mb-3">理想の顔</h3>
                    <div className="aspect-square rounded-xl overflow-hidden mb-3">
                      {idealImage.previewUrl && (
                        <img
                          src={idealImage.previewUrl}
                          alt="理想の顔"
                          className="w-full h-full object-cover"
                        />
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={handleChangeIdealImage}
                      className="w-full px-4 py-2 text-sm text-neutral-600 border border-neutral-300 rounded-full hover:bg-neutral-50 transition-colors"
                    >
                      変更する
                    </button>
                  </div>
                </div>

                {/* 現在の顔 */}
                <div className="w-full max-w-[280px]">
                  <div className="bg-white rounded-2xl shadow-elegant p-4">
                    <h3 className="font-serif text-lg text-neutral-800 text-center mb-3">現在の顔</h3>
                    <div className="aspect-square rounded-xl overflow-hidden mb-3">
                      {currentImage.previewUrl && (
                        <img
                          src={currentImage.previewUrl}
                          alt="現在の顔"
                          className="w-full h-full object-cover"
                        />
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={handleChangeCurrentImage}
                      className="w-full px-4 py-2 text-sm text-neutral-600 border border-neutral-300 rounded-full hover:bg-neutral-50 transition-colors"
                    >
                      変更する
                    </button>
                  </div>
                </div>
              </div>

              {/* 生成ボタン */}
              <button
                type="button"
                onClick={handleGenerate}
                disabled={isGenerating}
                className={`
                  w-full max-w-md px-8 py-4 text-base font-medium
                  rounded-full transition-all duration-300
                  focus:outline-none focus:ring-2 focus:ring-offset-2
                  ${
                    !isGenerating
                      ? 'bg-primary-700 text-white hover:bg-primary-800 hover:shadow-elegant focus:ring-primary-500'
                      : 'bg-neutral-200 text-neutral-400 cursor-not-allowed'
                  }
                `}
                data-testid="generate-button"
              >
                {isGenerating ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg
                      className="animate-spin h-5 w-5 text-white"
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
                  'シミュレーションを生成'
                )}
              </button>
            </div>
          )}

          {/* 注意書き */}
          <p className="mt-8 text-center text-sm text-neutral-400">
            ※ 顔写真は正面を向いた明るい写真をお使いください
          </p>

          {/* 利用規約・プライバシーポリシーリンク */}
          <div className="mt-6 flex justify-center gap-6 text-sm text-neutral-400">
            <a href="/terms" className="hover:text-primary-600 transition-colors">
              利用規約
            </a>
            <a href="/privacy" className="hover:text-primary-600 transition-colors">
              プライバシーポリシー
            </a>
          </div>
        </div>
      </main>

      <Footer />

      {/* 利用規約同意モーダル */}
      <TermsAgreementModal
        isOpen={isTermsModalOpen}
        onClose={handleTermsModalClose}
        onAgree={handleTermsAgree}
        testId="terms-agreement-modal"
      />
    </div>
  )
}
