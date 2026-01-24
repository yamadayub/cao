'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

export interface DeleteConfirmModalProps {
  /** モーダル表示状態 */
  isOpen: boolean
  /** 削除対象のID（表示用） */
  targetId?: string
  /** 削除処理中フラグ */
  isDeleting?: boolean
  /** 削除確定時のコールバック */
  onConfirm: () => void
  /** キャンセル時のコールバック */
  onCancel: () => void
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * 削除確認モーダルコンポーネント
 *
 * シミュレーション削除前の確認ダイアログ
 */
export function DeleteConfirmModal({
  isOpen,
  targetId,
  isDeleting = false,
  onConfirm,
  onCancel,
  testId,
}: DeleteConfirmModalProps) {
  const modalRef = useRef<HTMLDivElement>(null)
  const cancelButtonRef = useRef<HTMLButtonElement>(null)

  /**
   * ESCキーでモーダルを閉じる
   */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isDeleting) {
        onCancel()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, isDeleting, onCancel])

  /**
   * モーダル表示時にキャンセルボタンにフォーカス
   */
  useEffect(() => {
    if (isOpen && cancelButtonRef.current) {
      cancelButtonRef.current.focus()
    }
  }, [isOpen])

  /**
   * オーバーレイクリックでモーダルを閉じる
   */
  const handleOverlayClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget && !isDeleting) {
        onCancel()
      }
    },
    [isDeleting, onCancel]
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
      aria-labelledby="delete-confirm-title"
      data-testid={testId}
    >
      <div
        ref={modalRef}
        className="relative w-full max-w-md bg-white rounded-xl shadow-2xl p-6 animate-in fade-in zoom-in-95 duration-200"
        role="document"
      >
        {/* アイコン */}
        <div className="mx-auto w-12 h-12 flex items-center justify-center bg-red-100 rounded-full mb-4">
          <svg
            className="w-6 h-6 text-red-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </div>

        {/* タイトル */}
        <h2
          id="delete-confirm-title"
          className="text-xl font-bold text-gray-900 text-center mb-2"
          data-testid={testId ? `${testId}-title` : undefined}
        >
          シミュレーションを削除
        </h2>

        {/* 説明文 */}
        <p className="text-gray-600 text-center mb-6">
          このシミュレーションを削除しますか？
          <br />
          <span className="text-red-500 font-medium">この操作は取り消せません。</span>
        </p>

        {/* ボタン群 */}
        <div className="flex gap-3">
          <button
            ref={cancelButtonRef}
            type="button"
            onClick={onCancel}
            disabled={isDeleting}
            className="flex-1 px-6 py-3 text-base font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid={testId ? `${testId}-cancel-button` : undefined}
          >
            キャンセル
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isDeleting}
            className="flex-1 px-6 py-3 text-base font-semibold text-white bg-red-600 rounded-lg hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid={testId ? `${testId}-confirm-button` : undefined}
          >
            {isDeleting ? (
              <span className="flex items-center justify-center gap-2">
                <svg
                  className="w-5 h-5 animate-spin"
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
                削除中...
              </span>
            ) : (
              '削除'
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
