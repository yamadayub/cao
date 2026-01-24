'use client'

import { useCallback, useRef, useState } from 'react'
import {
  validateImageFormat,
  validateImageSize,
  type ImageValidationResult,
} from '@/lib/validation'

export interface ImageUploaderProps {
  /** アップローダーのラベル */
  label: string
  /** プレビュー画像のURL（Data URL） */
  previewUrl: string | null
  /** エラーメッセージ */
  error?: string
  /** ファイル選択時のコールバック */
  onFileSelect: (file: File, previewUrl: string) => void
  /** ファイル削除時のコールバック */
  onFileRemove: () => void
  /** バリデーションエラー時のコールバック */
  onValidationError: (error: string) => void
  /**
   * ファイル選択前のコールバック
   * falseを返すとファイル選択ダイアログが開かない
   * 利用規約同意チェックなどに使用
   */
  onClickBeforeSelect?: () => boolean
  /** 無効状態 */
  disabled?: boolean
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * 画像アップロードコンポーネント
 *
 * ドラッグ&ドロップとファイル選択ダイアログに対応
 * プレビュー表示と削除機能を提供
 */
export function ImageUploader({
  label,
  previewUrl,
  error,
  onFileSelect,
  onFileRemove,
  onValidationError,
  onClickBeforeSelect,
  disabled = false,
  testId,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  /**
   * ファイルを処理してバリデーション後にコールバックを呼び出す
   */
  const processFile = useCallback(
    (file: File) => {
      // 形式チェック
      const formatResult: ImageValidationResult = validateImageFormat(file)
      if (!formatResult.valid && formatResult.error) {
        onValidationError(formatResult.error.message)
        return
      }

      // サイズチェック
      const sizeResult: ImageValidationResult = validateImageSize(file)
      if (!sizeResult.valid && sizeResult.error) {
        onValidationError(sizeResult.error.message)
        return
      }

      // プレビュー生成
      const reader = new FileReader()
      reader.onload = (e) => {
        const dataUrl = e.target?.result as string
        onFileSelect(file, dataUrl)
      }
      reader.readAsDataURL(file)
    },
    [onFileSelect, onValidationError]
  )

  /**
   * ファイル選択ダイアログからの選択
   */
  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        processFile(file)
      }
      // 同じファイルを再選択できるようにリセット
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    },
    [processFile]
  )

  /**
   * ドラッグオーバー
   */
  const handleDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      if (!disabled) {
        setIsDragging(true)
      }
    },
    [disabled]
  )

  /**
   * ドラッグ離脱
   */
  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  /**
   * ドロップ
   */
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragging(false)

      if (disabled) return

      // 選択前のコールバックがあり、falseを返した場合はファイルを処理しない
      if (onClickBeforeSelect && !onClickBeforeSelect()) {
        return
      }

      const file = e.dataTransfer.files?.[0]
      if (file) {
        processFile(file)
      }
    },
    [disabled, onClickBeforeSelect, processFile]
  )

  /**
   * クリックでファイル選択ダイアログを開く
   */
  const handleClick = useCallback(() => {
    if (disabled) return

    // 選択前のコールバックがあり、falseを返した場合はダイアログを開かない
    if (onClickBeforeSelect && !onClickBeforeSelect()) {
      return
    }

    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }, [disabled, onClickBeforeSelect])

  /**
   * キーボード操作でのファイル選択
   */
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
        e.preventDefault()
        // 選択前のコールバックがあり、falseを返した場合はダイアログを開かない
        if (onClickBeforeSelect && !onClickBeforeSelect()) {
          return
        }
        fileInputRef.current?.click()
      }
    },
    [disabled, onClickBeforeSelect]
  )

  return (
    <div className="flex flex-col gap-3" data-testid={testId}>
      {/* ラベル */}
      <h3 className="text-lg font-semibold text-gray-800">{label}</h3>

      {/* ドロップエリア */}
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label={`${label}をアップロード`}
        aria-disabled={disabled}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative flex flex-col items-center justify-center
          w-full aspect-square max-w-[280px]
          border-2 border-dashed rounded-lg
          transition-all duration-200
          ${
            disabled
              ? 'bg-gray-100 border-gray-300 cursor-not-allowed'
              : isDragging
                ? 'bg-blue-50 border-blue-400 cursor-copy'
                : 'bg-gray-50 border-gray-300 hover:border-blue-400 hover:bg-blue-50 cursor-pointer'
          }
          ${error ? 'border-red-400 bg-red-50' : ''}
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
        `}
        data-testid={testId ? `${testId}-dropzone` : undefined}
      >
        {previewUrl ? (
          // プレビュー表示
          <div className="relative w-full h-full p-2">
            <img
              src={previewUrl}
              alt={`${label}のプレビュー`}
              className="w-full h-full object-cover rounded-md"
              data-testid={testId ? `${testId}-preview` : undefined}
            />
          </div>
        ) : (
          // プレースホルダー
          <div className="flex flex-col items-center gap-2 p-4 text-center">
            {/* アップロードアイコン */}
            <svg
              className={`w-12 h-12 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <p className="text-sm text-gray-600">
              {isDragging ? 'ここにドロップ' : 'ドラッグ&ドロップ'}
            </p>
            <p className="text-xs text-gray-500">または下のボタンで選択</p>
          </div>
        )}

        {/* 隠しファイルインプット */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleFileChange}
          disabled={disabled}
          className="sr-only"
          aria-label={`${label}を選択`}
          data-testid={testId ? `${testId}-input` : undefined}
        />
      </div>

      {/* ボタン群 */}
      <div className="flex gap-2">
        {!previewUrl ? (
          <button
            type="button"
            onClick={handleClick}
            disabled={disabled}
            className={`
              flex-1 px-4 py-2 text-sm font-medium
              rounded-md transition-colors
              ${
                disabled
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
              }
            `}
            data-testid={testId ? `${testId}-select-button` : undefined}
          >
            画像を選択
          </button>
        ) : (
          <button
            type="button"
            onClick={onFileRemove}
            disabled={disabled}
            className={`
              flex-1 px-4 py-2 text-sm font-medium
              rounded-md transition-colors
              ${
                disabled
                  ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                  : 'bg-red-600 text-white hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2'
              }
            `}
            data-testid={testId ? `${testId}-remove-button` : undefined}
          >
            削除
          </button>
        )}
      </div>

      {/* エラーメッセージ */}
      {error && (
        <p
          className="text-sm text-red-600"
          role="alert"
          data-testid={testId ? `${testId}-error` : undefined}
        >
          {error}
        </p>
      )}
    </div>
  )
}
