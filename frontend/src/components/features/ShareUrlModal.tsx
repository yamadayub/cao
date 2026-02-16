'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslations } from 'next-intl'

export interface ShareUrlModalProps {
  /** モーダル表示状態 */
  isOpen: boolean
  /** 共有URL */
  shareUrl: string
  /** モーダルを閉じるコールバック */
  onClose: () => void
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * 共有URLコピーモーダルコンポーネント
 *
 * 共有URL生成後に表示し、URLをクリップボードにコピーできる
 */
export function ShareUrlModal({
  isOpen,
  shareUrl,
  onClose,
  testId,
}: ShareUrlModalProps) {
  const t = useTranslations('modals')
  const [isCopied, setIsCopied] = useState(false)
  const modalRef = useRef<HTMLDivElement>(null)
  const copyButtonRef = useRef<HTMLButtonElement>(null)

  /**
   * ESCキーでモーダルを閉じる
   */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  /**
   * モーダル表示時にフォーカス
   */
  useEffect(() => {
    if (isOpen && copyButtonRef.current) {
      copyButtonRef.current.focus()
    }
  }, [isOpen])

  /**
   * コピー状態をリセット
   */
  useEffect(() => {
    if (!isOpen) {
      setIsCopied(false)
    }
  }, [isOpen])

  /**
   * URLをクリップボードにコピー
   */
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(shareUrl)
      setIsCopied(true)
      // 2秒後にコピー状態をリセット
      setTimeout(() => setIsCopied(false), 2000)
    } catch (error) {
      console.error('Failed to copy URL:', error)
      // フォールバック：テキスト選択
      const input = document.querySelector<HTMLInputElement>('[data-share-url-input]')
      if (input) {
        input.select()
        document.execCommand('copy')
        setIsCopied(true)
        setTimeout(() => setIsCopied(false), 2000)
      }
    }
  }, [shareUrl])

  /**
   * オーバーレイクリックでモーダルを閉じる
   */
  const handleOverlayClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) {
        onClose()
      }
    },
    [onClose]
  )

  if (!isOpen) {
    return null
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="share-url-title"
      data-testid={testId}
    >
      <div
        ref={modalRef}
        className="relative w-full max-w-md bg-white rounded-xl shadow-2xl p-6 animate-in fade-in zoom-in-95 duration-200"
        role="document"
      >
        {/* 閉じるボタン */}
        <button
          type="button"
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-gray-400 hover:text-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-full"
          aria-label={t('common.close')}
          data-testid={testId ? `${testId}-close` : undefined}
        >
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
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        {/* コンテンツ */}
        <div className="text-center">
          {/* アイコン */}
          <div className="mx-auto w-12 h-12 flex items-center justify-center bg-green-100 rounded-full mb-4">
            <svg
              className="w-6 h-6 text-green-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"
              />
            </svg>
          </div>

          {/* タイトル */}
          <h2
            id="share-url-title"
            className="text-xl font-bold text-gray-900 mb-2"
            data-testid={testId ? `${testId}-title` : undefined}
          >
            {t('share.title')}
          </h2>

          {/* 説明文 */}
          <p className="text-gray-600 mb-4">
            {t('share.description')}
          </p>

          {/* URL表示 */}
          <div className="relative mb-4">
            <input
              type="text"
              value={shareUrl}
              readOnly
              data-share-url-input
              className="w-full px-4 py-3 pr-12 text-sm text-gray-700 bg-gray-100 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              onClick={(e) => (e.target as HTMLInputElement).select()}
              data-testid={testId ? `${testId}-url-input` : undefined}
            />
          </div>

          {/* ボタン群 */}
          <div className="flex flex-col gap-3">
            <button
              ref={copyButtonRef}
              type="button"
              onClick={handleCopy}
              className={`
                w-full px-6 py-3 text-base font-semibold rounded-lg
                focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200
                ${
                  isCopied
                    ? 'bg-green-600 text-white focus:ring-green-500'
                    : 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
                }
              `}
              data-testid={testId ? `${testId}-copy-button` : undefined}
            >
              {isCopied ? (
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
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                  {t('share.copied')}
                </span>
              ) : (
                t('share.copyButton')
              )}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="w-full px-6 py-3 text-base font-medium text-gray-600 hover:text-gray-800 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded-lg"
              data-testid={testId ? `${testId}-close-button` : undefined}
            >
              {t('common.close')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
