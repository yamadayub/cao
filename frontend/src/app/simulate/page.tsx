'use client'

import { useCallback, useState } from 'react'
import { useRouter } from 'next/navigation'
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
      <main className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-gray-500">読み込み中...</div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gray-50">
      {/* ヘッダー */}
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-900">Cao</h1>
          <nav className="flex gap-4">
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
          </nav>
        </div>
      </header>

      {/* メインコンテンツ */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* ページタイトル */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-900">
            シミュレーション作成
          </h2>
          <p className="mt-2 text-gray-600">
            現在の顔と理想の顔をアップロードして、シミュレーションを生成します
          </p>
        </div>

        {/* 利用規約同意バナー（未同意の場合） */}
        {!hasAgreed && (
          <div
            className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg"
            data-testid="terms-agreement-banner"
          >
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0"
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
                <p className="text-sm text-blue-800">
                  画像をアップロードするには、
                  <button
                    type="button"
                    onClick={() => setIsTermsModalOpen(true)}
                    className="font-medium underline hover:text-blue-900"
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
          className="flex flex-col md:flex-row gap-8 justify-center items-start"
          data-testid="upload-area"
        >
          {/* 現在の顔 */}
          <div className="flex-1 max-w-[320px] bg-white rounded-xl shadow-sm p-6">
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
          <div className="flex-1 max-w-[320px] bg-white rounded-xl shadow-sm p-6">
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
        <div className="mt-8 flex justify-center">
          <button
            type="button"
            onClick={handleGenerate}
            disabled={!canGenerate}
            className={`
              w-full max-w-md px-8 py-4 text-lg font-semibold
              rounded-lg transition-all duration-200
              focus:outline-none focus:ring-2 focus:ring-offset-2
              ${
                canGenerate
                  ? 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
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
        <p className="mt-6 text-center text-sm text-gray-500">
          ※ 顔写真は正面を向いた明るい写真をお使いください
        </p>

        {/* 利用規約・プライバシーポリシーリンク */}
        <div className="mt-8 flex justify-center gap-6 text-sm text-gray-500">
          <a href="/terms" className="hover:text-gray-700 underline">
            利用規約
          </a>
          <a href="/privacy" className="hover:text-gray-700 underline">
            プライバシーポリシー
          </a>
        </div>
      </div>

      {/* 利用規約同意モーダル */}
      <TermsAgreementModal
        isOpen={isTermsModalOpen}
        onClose={handleTermsModalClose}
        onAgree={handleTermsAgree}
        testId="terms-agreement-modal"
      />
    </main>
  )
}
