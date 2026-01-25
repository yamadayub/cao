'use client'

import { useCallback, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { ImageUploader } from '@/components/features/ImageUploader'
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
 * エラーのステート型
 */
interface ErrorState {
  current?: string
  ideal?: string
}

/**
 * シミュレーション作成画面 (SCR-002)
 *
 * 参照: functional-spec.md セクション 3.3
 * 参照: business-spec.md UC-001, UC-002, UC-003
 */
export default function SimulatePage() {
  const router = useRouter()
  const { hasAgreed, isLoading: isLoadingTerms, agree } = useTermsAgreement()

  // 現在の顔画像の状態
  const [currentImage, setCurrentImage] = useState<ImageState>({
    file: null,
    previewUrl: null,
  })

  // 理想の顔画像の状態
  const [idealImage, setIdealImage] = useState<ImageState>({
    file: null,
    previewUrl: null,
  })

  // エラー状態
  const [errors, setErrors] = useState<ErrorState>({})

  // 生成中フラグ
  const [isGenerating, setIsGenerating] = useState(false)

  // 利用規約同意モーダル表示状態
  const [isTermsModalOpen, setIsTermsModalOpen] = useState(false)

  // 利用規約同意後に実行するアクション（'current' | 'ideal' | null）
  const [pendingUploadType, setPendingUploadType] = useState<'current' | 'ideal' | null>(null)

  /**
   * 画像アップロードを試みる（未同意の場合はモーダル表示）
   * @returns boolean - 同意済みの場合はtrue、未同意の場合はfalse
   */
  const attemptUpload = useCallback(
    (uploadType: 'current' | 'ideal') => {
      if (!hasAgreed) {
        setPendingUploadType(uploadType)
        setIsTermsModalOpen(true)
        return false
      }
      return true
    },
    [hasAgreed]
  )

  /**
   * 現在の顔画像選択前のチェック
   */
  const handleCurrentBeforeSelect = useCallback(() => {
    return attemptUpload('current')
  }, [attemptUpload])

  /**
   * 理想の顔画像選択前のチェック
   */
  const handleIdealBeforeSelect = useCallback(() => {
    return attemptUpload('ideal')
  }, [attemptUpload])

  /**
   * 利用規約同意モーダルを閉じる
   */
  const handleTermsModalClose = useCallback(() => {
    setIsTermsModalOpen(false)
    setPendingUploadType(null)
  }, [])

  /**
   * 利用規約に同意する
   */
  const handleTermsAgree = useCallback(() => {
    agree()
    setIsTermsModalOpen(false)
    // 同意後、保留中のアップロードがあればそのフィールドにフォーカスを移す
    // （実際のファイル選択ダイアログは開かないが、UIの準備は整う）
    setPendingUploadType(null)
  }, [agree])

  /**
   * 現在の顔画像選択ハンドラ
   */
  const handleCurrentFileSelect = useCallback(
    (file: File, previewUrl: string) => {
      setCurrentImage({ file, previewUrl })
      setErrors((prev) => ({ ...prev, current: undefined }))
    },
    []
  )

  /**
   * 現在の顔画像削除ハンドラ
   */
  const handleCurrentFileRemove = useCallback(() => {
    setCurrentImage({ file: null, previewUrl: null })
    setErrors((prev) => ({ ...prev, current: undefined }))
  }, [])

  /**
   * 現在の顔画像バリデーションエラーハンドラ
   */
  const handleCurrentValidationError = useCallback((error: string) => {
    setErrors((prev) => ({ ...prev, current: error }))
  }, [])

  /**
   * 理想の顔画像選択ハンドラ
   */
  const handleIdealFileSelect = useCallback(
    (file: File, previewUrl: string) => {
      setIdealImage({ file, previewUrl })
      setErrors((prev) => ({ ...prev, ideal: undefined }))
    },
    []
  )

  /**
   * 理想の顔画像削除ハンドラ
   */
  const handleIdealFileRemove = useCallback(() => {
    setIdealImage({ file: null, previewUrl: null })
    setErrors((prev) => ({ ...prev, ideal: undefined }))
  }, [])

  /**
   * 理想の顔画像バリデーションエラーハンドラ
   */
  const handleIdealValidationError = useCallback((error: string) => {
    setErrors((prev) => ({ ...prev, ideal: error }))
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
      // 現在は仮の遅延を入れて結果画面に遷移
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

  // 両方の画像がアップロードされているかチェック
  const canGenerate =
    currentImage.file !== null && idealImage.file !== null && !isGenerating

  // ローディング中は簡易表示
  if (isLoadingTerms) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="w-8 h-8 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin"></div>
      </div>
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
              現在の顔と理想の顔をアップロードして、シミュレーションを生成します
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

          {/* アップロードエリア */}
          <div
            className="flex flex-col md:flex-row gap-4 md:gap-8 justify-center items-center"
            data-testid="upload-area"
          >
            {/* 現在の顔 */}
            <div className="w-full max-w-[300px] bg-white rounded-2xl shadow-elegant p-5 md:p-6">
              <ImageUploader
                label="現在の顔"
                previewUrl={currentImage.previewUrl}
                error={errors.current}
                onFileSelect={handleCurrentFileSelect}
                onFileRemove={handleCurrentFileRemove}
                onValidationError={handleCurrentValidationError}
                onClickBeforeSelect={handleCurrentBeforeSelect}
                disabled={isGenerating}
                testId="current-image"
              />
            </div>

            {/* 理想の顔 */}
            <div className="w-full max-w-[300px] bg-white rounded-2xl shadow-elegant p-5 md:p-6">
              <ImageUploader
                label="理想の顔"
                previewUrl={idealImage.previewUrl}
                error={errors.ideal}
                onFileSelect={handleIdealFileSelect}
                onFileRemove={handleIdealFileRemove}
                onValidationError={handleIdealValidationError}
                onClickBeforeSelect={handleIdealBeforeSelect}
                disabled={isGenerating}
                testId="ideal-image"
              />
            </div>
          </div>

          {/* 生成ボタン */}
          <div className="mt-8 md:mt-12 flex justify-center">
            <button
              type="button"
              onClick={handleGenerate}
              disabled={!canGenerate}
              className={`
                w-full max-w-md px-8 py-4 text-base font-medium
                rounded-full transition-all duration-300
                focus:outline-none focus:ring-2 focus:ring-offset-2
                ${
                  canGenerate
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

          {/* 注意書き */}
          <p className="mt-6 text-center text-sm text-neutral-400">
            ※ 顔写真は正面を向いた明るい写真をお使いください
          </p>

          {/* 利用規約・プライバシーポリシーリンク */}
          <div className="mt-8 flex justify-center gap-6 text-sm text-neutral-400">
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
