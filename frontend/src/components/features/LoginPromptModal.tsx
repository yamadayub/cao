'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

export interface LoginPromptModalProps {
  /** モーダル表示状態 */
  isOpen: boolean
  /** モーダルを閉じるコールバック */
  onClose: () => void
  /** ログインボタンクリック時のコールバック（フォールバック用） */
  onLogin: () => void
  /** モーダルのタイトル */
  title?: string
  /** モーダルの説明文 */
  description?: string
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * ログイン誘導モーダルコンポーネント
 *
 * 未認証ユーザーが保存や共有機能を使用しようとした際に表示
 */
export function LoginPromptModal({
  isOpen,
  onClose,
  onLogin,
  title = '保存するにはログインが必要です',
  description = 'シミュレーション結果を保存・共有するにはログインが必要です。',
  testId,
}: LoginPromptModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const [SignInButton, setSignInButton] = useState<React.ComponentType<{
    mode: string
    children: React.ReactNode
  }> | null>(null)

  /**
   * Clerkのコンポーネントを動的にロード
   */
  useEffect(() => {
    const loadClerk = async () => {
      try {
        const clerk = await import('@clerk/nextjs')
        setSignInButton(() => clerk.SignInButton as React.ComponentType<{
          mode: string
          children: React.ReactNode
        }>)
      } catch {
        console.warn('Clerk is not available')
      }
    }
    loadClerk()
  }, [])

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
   * モーダル表示時にフォーカスをトラップ
   */
  useEffect(() => {
    if (isOpen && closeButtonRef.current) {
      closeButtonRef.current.focus()
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

  if (!isOpen) {
    return null
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
      aria-labelledby="login-prompt-title"
      data-testid={testId}
    >
      <div
        ref={modalRef}
        className="relative w-full max-w-md bg-white rounded-xl shadow-2xl p-6 animate-in fade-in zoom-in-95 duration-200"
        role="document"
      >
        {/* 閉じるボタン */}
        <button
          ref={closeButtonRef}
          type="button"
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-gray-400 hover:text-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-full"
          aria-label="閉じる"
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
          <div className="mx-auto w-12 h-12 flex items-center justify-center bg-blue-100 rounded-full mb-4">
            <svg
              className="w-6 h-6 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
              />
            </svg>
          </div>

          {/* タイトル */}
          <h2
            id="login-prompt-title"
            className="text-xl font-bold text-gray-900 mb-2"
            data-testid={testId ? `${testId}-title` : undefined}
          >
            {title}
          </h2>

          {/* 説明文 */}
          <p
            className="text-gray-600 mb-6"
            data-testid={testId ? `${testId}-description` : undefined}
          >
            {description}
          </p>

          {/* ボタン群 */}
          <div className="flex flex-col gap-3">
            {SignInButton ? (
              <SignInButton mode="modal">
                <button
                  type="button"
                  onClick={onClose}
                  className="w-full px-6 py-3 text-base font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
                  data-testid={testId ? `${testId}-login-button` : undefined}
                >
                  ログインする
                </button>
              </SignInButton>
            ) : (
              <button
                type="button"
                onClick={onLogin}
                className="w-full px-6 py-3 text-base font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
                data-testid={testId ? `${testId}-login-button` : undefined}
              >
                ログインする
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="w-full px-6 py-3 text-base font-medium text-gray-600 hover:text-gray-800 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded-lg"
              data-testid={testId ? `${testId}-cancel-button` : undefined}
            >
              今はログインしない
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
