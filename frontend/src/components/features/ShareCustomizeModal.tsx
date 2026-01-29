'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import type { ShareTemplate } from '@/lib/api/types'

export interface ShareCustomizeModalProps {
  /** モーダル表示状態 */
  isOpen: boolean
  /** モーダルを閉じるコールバック */
  onClose: () => void
  /** ソース画像（Base64） */
  sourceImage: string
  /** 結果画像（Base64） */
  resultImage: string
  /** シェア作成コールバック */
  onCreateShare: (data: {
    template: ShareTemplate
    caption: string
  }) => Promise<void>
  /** シェアURL（作成後） */
  shareUrl?: string
  /** シェア画像URL（作成後） */
  shareImageUrl?: string
  /** 作成中フラグ */
  isCreating: boolean
  /** テスト用のdata-testid */
  testId?: string
}

const TEMPLATE_OPTIONS: Array<{ value: ShareTemplate; label: string; icon: React.ReactNode }> = [
  {
    value: 'before_after',
    label: '比較',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7" />
      </svg>
    ),
  },
  {
    value: 'single',
    label: '単体',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
  },
]

const PRIVACY_WARNINGS = [
  'Anyone can view shared images',
  'Shared images may not be deletable',
  'Do not share photos of others without permission',
]

/**
 * シェアカスタマイズモーダル
 *
 * UC-014: シェア画像のカスタマイズ
 * UC-015: SNSへのシェア
 */
export function ShareCustomizeModal({
  isOpen,
  onClose,
  sourceImage,
  resultImage,
  onCreateShare,
  shareUrl,
  shareImageUrl,
  isCreating,
  testId,
}: ShareCustomizeModalProps) {
  const [template, setTemplate] = useState<ShareTemplate>('before_after')
  const [caption, setCaption] = useState('')
  const [privacyAgreed, setPrivacyAgreed] = useState(false)
  const modalRef = useRef<HTMLDivElement>(null)

  // リセット
  useEffect(() => {
    if (!isOpen) {
      setTemplate('before_after')
      setCaption('')
      setPrivacyAgreed(false)
    }
  }, [isOpen])

  // ESCキーで閉じる
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  const handleSubmit = useCallback(async () => {
    await onCreateShare({ template, caption })
  }, [template, caption, onCreateShare])

  const handleCaptionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCaption(e.target.value.slice(0, 140))
  }, [])

  const handleOverlayClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) {
        onClose()
      }
    },
    [onClose]
  )

  const handleTwitterShare = useCallback(() => {
    if (!shareUrl) return
    const twitterUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(caption)}&url=${encodeURIComponent(shareUrl)}`
    window.open(twitterUrl, '_blank', 'noopener,noreferrer')
  }, [shareUrl, caption])

  const handleLineShare = useCallback(() => {
    if (!shareUrl) return
    const lineUrl = `https://social-plugins.line.me/lineit/share?url=${encodeURIComponent(shareUrl)}`
    window.open(lineUrl, '_blank', 'noopener,noreferrer')
  }, [shareUrl])

  const handleDownload = useCallback(() => {
    if (!shareImageUrl) return
    const link = document.createElement('a')
    link.href = shareImageUrl
    link.download = 'cao-share.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [shareImageUrl])

  const handleInstagramShare = useCallback(() => {
    if (shareImageUrl) {
      handleDownload()
    }
  }, [shareImageUrl, handleDownload])

  const handleCopyUrl = useCallback(async () => {
    if (!shareUrl) return
    try {
      await navigator.clipboard.writeText(shareUrl)
    } catch {
      // フォールバック
      const textArea = document.createElement('textarea')
      textArea.value = shareUrl
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand('copy')
      document.body.removeChild(textArea)
    }
  }, [shareUrl])

  const isShareReady = !!shareUrl && !!shareImageUrl

  if (!isOpen) {
    return null
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
      onClick={handleOverlayClick}
      role="dialog"
      aria-modal="true"
      data-testid={testId}
    >
      <div
        ref={modalRef}
        className="relative w-full max-w-sm bg-white rounded-xl shadow-xl overflow-hidden animate-in fade-in zoom-in-95 duration-200"
        role="document"
      >
        {/* ヘッダー */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <h2 className="text-base font-semibold text-gray-800 tracking-tight">
            {!isShareReady ? 'Create Share Image' : 'Share'}
          </h2>
          <button
            type="button"
            onClick={onClose}
            data-testid={testId ? `${testId}-close` : undefined}
            className="p-2 -mr-2 text-gray-400 hover:text-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-full"
            aria-label="閉じる"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* コンテンツ */}
        <div className="p-5 max-h-[75vh] overflow-y-auto">
          {!isShareReady ? (
            /* 作成画面 */
            <div className="space-y-5">
              {/* プレビュー */}
              <div
                data-testid={testId ? `${testId}-preview` : undefined}
                className="rounded-xl overflow-hidden bg-gray-50 border border-gray-100"
              >
                <div className="flex">
                  <div className="flex-1 p-2">
                    <img
                      src={sourceImage}
                      alt="変更前"
                      className="w-full aspect-square object-cover rounded-lg"
                    />
                    <p className="text-xs text-gray-500 text-center mt-1">Before</p>
                  </div>
                  <div className="flex-1 p-2">
                    <img
                      src={resultImage}
                      alt="変更後"
                      className="w-full aspect-square object-cover rounded-lg"
                    />
                    <p className="text-xs text-gray-500 text-center mt-1">After</p>
                  </div>
                </div>
              </div>

              {/* テンプレート選択 */}
              <div data-testid={testId ? `${testId}-templates` : undefined}>
                <label className="block text-xs font-medium text-gray-500 mb-2 tracking-wide uppercase">
                  Layout
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {TEMPLATE_OPTIONS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setTemplate(option.value)}
                      data-testid={testId ? `${testId}-template-${option.value}` : undefined}
                      data-selected={template === option.value}
                      className={`
                        py-2.5 px-4 rounded-lg border transition-all
                        flex items-center justify-center gap-2
                        focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-1
                        ${template === option.value
                          ? 'border-gray-900 bg-gray-900 text-white'
                          : 'border-gray-200 hover:border-gray-300 text-gray-500 hover:text-gray-700'
                        }
                      `}
                    >
                      {option.icon}
                      <span className="text-sm font-medium">{option.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* キャプション入力 */}
              <div>
                <label htmlFor="share-caption" className="block text-xs font-medium text-gray-500 mb-2 tracking-wide uppercase">
                  Caption
                  <span className="text-gray-400 font-normal normal-case ml-1">（optional）</span>
                </label>
                <textarea
                  id="share-caption"
                  value={caption}
                  onChange={handleCaptionChange}
                  maxLength={140}
                  placeholder="Add a message..."
                  rows={2}
                  data-testid={testId ? `${testId}-caption` : undefined}
                  className="w-full px-3 py-2.5 border border-gray-200 rounded-lg resize-none text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent placeholder:text-gray-400"
                />
                <div className="text-right mt-1">
                  <span
                    data-testid={testId ? `${testId}-caption-count` : undefined}
                    className={`text-xs ${caption.length >= 120 ? 'text-amber-600' : 'text-gray-400'}`}
                  >
                    {caption.length}/140
                  </span>
                </div>
              </div>

              {/* 作成ボタン */}
              <button
                type="button"
                onClick={handleSubmit}
                disabled={isCreating}
                data-testid={testId ? `${testId}-create-button` : undefined}
                className={`
                  w-full py-3 px-4 rounded-lg font-medium text-sm
                  transition-all duration-200
                  focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500
                  ${isCreating
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-gray-900 text-white hover:bg-gray-800 active:scale-[0.98]'
                  }
                `}
              >
                {isCreating ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Creating...
                  </span>
                ) : (
                  'Create Preview'
                )}
              </button>
            </div>
          ) : (
            /* シェア画面（プレビュー付き） */
            <div className="space-y-5">
              {/* シェア画像プレビュー */}
              <div
                data-testid={testId ? `${testId}-share-preview` : undefined}
                className="rounded-xl overflow-hidden bg-gray-50 border border-gray-100"
              >
                <img
                  src={shareImageUrl}
                  alt="シェア画像プレビュー"
                  className="w-full"
                />
              </div>

              {/* プライバシー警告 */}
              <div
                data-testid={testId ? `${testId}-privacy-warning` : undefined}
                className="p-3 bg-gray-50 border border-gray-100 rounded-lg"
              >
                <ul className="space-y-1 text-xs text-gray-500 mb-2.5">
                  {PRIVACY_WARNINGS.map((warning, index) => (
                    <li key={index} className="flex items-start gap-1.5">
                      <span className="text-gray-400">•</span>
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={privacyAgreed}
                    onChange={(e) => setPrivacyAgreed(e.target.checked)}
                    data-testid={testId ? `${testId}-privacy-checkbox` : undefined}
                    className="w-3.5 h-3.5 text-gray-900 rounded border-gray-300 focus:ring-gray-500"
                  />
                  <span className="text-xs font-medium text-gray-600">I understand</span>
                </label>
              </div>

              {/* シェアボタン群（アイコンのみ） */}
              <div data-testid={testId ? `${testId}-share-buttons` : undefined}>
                <p className="text-xs font-medium text-gray-500 mb-3 tracking-wide uppercase">Share to</p>
                <div className="flex justify-center gap-3">
                  {/* X (Twitter) */}
                  <button
                    type="button"
                    onClick={handleTwitterShare}
                    disabled={!privacyAgreed}
                    data-testid={testId ? `${testId}-share-twitter` : undefined}
                    title="X"
                    className={`
                      w-12 h-12 flex items-center justify-center rounded-full
                      transition-all duration-200
                      ${privacyAgreed
                        ? 'bg-gray-900 text-white hover:bg-gray-800 active:scale-95'
                        : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                      }
                    `}
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                    </svg>
                  </button>

                  {/* LINE */}
                  <button
                    type="button"
                    onClick={handleLineShare}
                    disabled={!privacyAgreed}
                    data-testid={testId ? `${testId}-share-line` : undefined}
                    title="LINE"
                    className={`
                      w-12 h-12 flex items-center justify-center rounded-full
                      transition-all duration-200
                      ${privacyAgreed
                        ? 'bg-[#06C755] text-white hover:bg-[#05b34d] active:scale-95'
                        : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                      }
                    `}
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M19.365 9.863c.349 0 .63.285.63.631 0 .345-.281.63-.63.63H17.61v1.125h1.755c.349 0 .63.283.63.63 0 .344-.281.629-.63.629h-2.386a.63.63 0 0 1-.63-.629V8.108a.63.63 0 0 1 .63-.63h2.386c.349 0 .63.285.63.63 0 .349-.281.63-.63.63H17.61v1.125h1.755zm-3.855 3.016a.63.63 0 0 1-.63.63.629.629 0 0 1-.511-.26l-2.4-3.255v2.885c0 .349-.283.63-.63.63a.63.63 0 0 1-.63-.63V8.108a.63.63 0 0 1 .63-.63c.2 0 .385.098.5.26l2.415 3.254V8.108c0-.345.285-.63.63-.63s.626.285.626.63v4.771zm-5.741 0a.63.63 0 0 1-.63.63.63.63 0 0 1-.63-.63V8.108a.63.63 0 0 1 .63-.63c.349 0 .63.285.63.63v4.771zm-2.466.63H4.917a.63.63 0 0 1-.63-.63V8.108a.63.63 0 0 1 .63-.63c.349 0 .63.285.63.63v4.141h1.756c.348 0 .629.283.629.63 0 .344-.281.63-.629.63M24 10.314C24 4.943 18.615.572 12 .572S0 4.943 0 10.314c0 4.811 4.27 8.842 10.035 9.608.391.082.923.258 1.058.59.12.301.079.766.038 1.08l-.164 1.02c-.045.301-.24 1.186 1.049.645 1.291-.539 6.916-4.078 9.436-6.975C23.176 14.393 24 12.458 24 10.314" />
                    </svg>
                  </button>

                  {/* Instagram */}
                  <button
                    type="button"
                    onClick={handleInstagramShare}
                    disabled={!privacyAgreed}
                    data-testid={testId ? `${testId}-share-instagram` : undefined}
                    title="Instagram"
                    className={`
                      w-12 h-12 flex items-center justify-center rounded-full
                      transition-all duration-200
                      ${privacyAgreed
                        ? 'bg-gradient-to-br from-purple-600 via-pink-500 to-orange-400 text-white hover:opacity-90 active:scale-95'
                        : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                      }
                    `}
                  >
                    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z" />
                    </svg>
                  </button>

                  {/* ダウンロード */}
                  <button
                    type="button"
                    onClick={handleDownload}
                    disabled={!privacyAgreed}
                    data-testid={testId ? `${testId}-download` : undefined}
                    title="Download"
                    className={`
                      w-12 h-12 flex items-center justify-center rounded-full
                      transition-all duration-200
                      ${privacyAgreed
                        ? 'bg-gray-100 text-gray-600 hover:bg-gray-200 active:scale-95'
                        : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                      }
                    `}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* URLコピー */}
              <div className="pt-2">
                <button
                  type="button"
                  onClick={handleCopyUrl}
                  disabled={!privacyAgreed}
                  data-testid={testId ? `${testId}-copy-url` : undefined}
                  className={`
                    w-full py-2.5 px-4 rounded-lg text-sm font-medium
                    flex items-center justify-center gap-2
                    transition-all duration-200
                    ${privacyAgreed
                      ? 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                      : 'bg-gray-50 text-gray-300 cursor-not-allowed border border-gray-100'
                    }
                  `}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Copy URL
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
