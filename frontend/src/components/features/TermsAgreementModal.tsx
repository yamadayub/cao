'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from '@/i18n/navigation'
import { useTranslations } from 'next-intl'

export interface TermsAgreementModalProps {
  /** モーダル表示状態 */
  isOpen: boolean
  /** モーダルを閉じるコールバック（同意せずに閉じる） */
  onClose: () => void
  /** 同意ボタンクリック時のコールバック */
  onAgree: () => void
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * 利用規約同意モーダルコンポーネント
 *
 * 画像アップロード前に表示し、利用規約への同意を求める
 * 参照: functional-spec.md セクション 3.3 利用規約同意モーダル
 */
export function TermsAgreementModal({
  isOpen,
  onClose,
  onAgree,
  testId = 'terms-agreement-modal',
}: TermsAgreementModalProps) {
  const t = useTranslations('modals')
  const modalRef = useRef<HTMLDivElement>(null)
  const agreeButtonRef = useRef<HTMLButtonElement>(null)
  const [isChecked, setIsChecked] = useState(false)

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
   * モーダル表示時にフォーカスを設定
   */
  useEffect(() => {
    if (isOpen && agreeButtonRef.current) {
      agreeButtonRef.current.focus()
    }
    // モーダルが開いたらチェック状態をリセット
    if (isOpen) {
      setIsChecked(false)
    }
  }, [isOpen])

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

  /**
   * チェックボックスの変更ハンドラ
   */
  const handleCheckboxChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setIsChecked(e.target.checked)
  }, [])

  /**
   * 同意ボタンクリックハンドラ
   */
  const handleAgreeClick = useCallback(() => {
    if (isChecked) {
      onAgree()
    }
  }, [isChecked, onAgree])

  if (!isOpen) {
    return null
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="terms-agreement-title"
      data-testid={testId}
    >
      <div
        ref={modalRef}
        className="relative w-full max-w-md bg-white rounded-xl shadow-2xl p-6 animate-in fade-in zoom-in-95 duration-200"
        role="document"
      >
        {/* タイトル */}
        <h2
          id="terms-agreement-title"
          className="text-xl font-bold text-gray-900 text-center mb-4"
          data-testid={`${testId}-title`}
        >
          {t('terms.title')}
        </h2>

        {/* 説明文 */}
        <p className="text-gray-600 text-center mb-6" dangerouslySetInnerHTML={{ __html: t('terms.description') }} />

        {/* チェックボックス */}
        <div className="mb-4">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={isChecked}
              onChange={handleCheckboxChange}
              className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
              data-testid={`${testId}-checkbox`}
            />
            <span className="text-gray-700">
              <Link
                href="/terms"
                target="_blank"
                className="text-blue-600 hover:text-blue-800 underline"
                data-testid={`${testId}-terms-link`}
              >
                {t('terms.termsLink')}
              </Link>
              {t('terms.agreeToTerms')}
            </span>
          </label>
        </div>

        {/* 利用規約を読むリンク */}
        <div className="mb-6 text-center">
          <Link
            href="/terms"
            target="_blank"
            className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 transition-colors"
            data-testid={`${testId}-read-terms-link`}
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
              />
            </svg>
            {t('terms.readTerms')}
          </Link>
        </div>

        {/* 同意ボタン */}
        <button
          ref={agreeButtonRef}
          type="button"
          onClick={handleAgreeClick}
          disabled={!isChecked}
          className={`
            w-full px-6 py-3 text-base font-semibold
            rounded-lg transition-all duration-200
            focus:outline-none focus:ring-2 focus:ring-offset-2
            ${
              isChecked
                ? 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }
          `}
          data-testid={`${testId}-agree-button`}
        >
          {t('terms.agreeButton')}
        </button>
      </div>
    </div>
  )
}
