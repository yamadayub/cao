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

const TEMPLATE_OPTIONS: Array<{ value: ShareTemplate; label: string; description: string }> = [
  { value: 'before_after', label: 'Before/After', description: '変化がわかりやすい比較画像' },
  { value: 'single', label: '単体', description: '結果のみを表示' },
  { value: 'parts_highlight', label: 'パーツハイライト', description: '変更箇所を強調' },
]

const PRIVACY_WARNINGS = [
  'シェアした画像は誰でも閲覧できます',
  '一度シェアした画像は削除できない場合があります',
  '他人の写真を無断でシェアしないでください',
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
  const [showShareOptions, setShowShareOptions] = useState(false)
  const [privacyAgreed, setPrivacyAgreed] = useState(false)
  const modalRef = useRef<HTMLDivElement>(null)

  // リセット
  useEffect(() => {
    if (!isOpen) {
      setTemplate('before_after')
      setCaption('')
      setShowShareOptions(false)
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
    setShowShareOptions(true)
  }, [template, caption, onCreateShare])

  const handleCaptionChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    // 140文字以内に制限
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

  const handleInstagramShare = useCallback(() => {
    // Instagramはネイティブシェアが必要
    // 画像をダウンロードしてからシェアを促す
    if (shareImageUrl) {
      handleDownload()
      alert('画像を保存しました。Instagramアプリで共有してください。')
    }
  }, [shareImageUrl])

  const handleDownload = useCallback(() => {
    if (!shareImageUrl) return
    const link = document.createElement('a')
    link.href = shareImageUrl
    link.download = 'cao-share.png'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [shareImageUrl])

  const captionError = caption.length > 140 ? 'キャプションは140文字以内で入力してください' : ''
  const isSubmitDisabled = isCreating || caption.length > 140

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
        className="relative w-full max-w-lg bg-white rounded-xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200"
        role="document"
      >
        {/* ヘッダー */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-bold text-gray-900">
            {!showShareOptions ? 'シェア画像を作成' : 'シェア先を選択'}
          </h2>
          <button
            type="button"
            onClick={onClose}
            data-testid={testId ? `${testId}-close` : undefined}
            className="p-1 text-gray-400 hover:text-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-full"
            aria-label="閉じる"
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
        </div>

        {/* コンテンツ */}
        <div className="p-4 max-h-[70vh] overflow-y-auto">
          {!showShareOptions ? (
            <>
              {/* プレビュー */}
              <div
                data-testid={testId ? `${testId}-preview` : undefined}
                className="mb-4 rounded-lg overflow-hidden bg-gray-100"
              >
                <div className="flex gap-2 p-2">
                  <img
                    src={sourceImage}
                    alt="変更前"
                    className="w-1/2 aspect-square object-cover rounded"
                  />
                  <img
                    src={resultImage}
                    alt="変更後"
                    className="w-1/2 aspect-square object-cover rounded"
                  />
                </div>
              </div>

              {/* テンプレート選択 */}
              <div
                data-testid={testId ? `${testId}-templates` : undefined}
                className="mb-4"
              >
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  テンプレート
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {TEMPLATE_OPTIONS.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setTemplate(option.value)}
                      data-testid={testId ? `${testId}-template-${option.value}` : undefined}
                      data-selected={template === option.value}
                      className={`
                        p-3 rounded-lg border-2 text-center transition-all
                        focus:outline-none focus:ring-2 focus:ring-blue-500
                        ${template === option.value
                          ? 'border-blue-600 bg-blue-50 text-blue-700'
                          : 'border-gray-200 hover:border-gray-300 text-gray-700'
                        }
                      `}
                    >
                      <div className="text-sm font-medium">{option.label}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* キャプション入力 */}
              <div className="mb-4">
                <label
                  htmlFor="share-caption"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  キャプション（任意）
                </label>
                <textarea
                  id="share-caption"
                  value={caption}
                  onChange={handleCaptionChange}
                  maxLength={140}
                  placeholder="キャプション（任意）"
                  rows={3}
                  data-testid={testId ? `${testId}-caption` : undefined}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <div className="flex justify-between items-center mt-1">
                  <span
                    data-testid={testId ? `${testId}-caption-count` : undefined}
                    className={`text-xs ${caption.length >= 120 ? 'text-amber-600' : 'text-gray-500'}`}
                  >
                    {caption.length}/140
                  </span>
                  {captionError && (
                    <span
                      data-testid={testId ? `${testId}-caption-error` : undefined}
                      role="alert"
                      className="text-xs text-red-600"
                    >
                      {captionError}
                    </span>
                  )}
                </div>
              </div>

              {/* 作成ボタン */}
              <button
                type="button"
                onClick={handleSubmit}
                disabled={isSubmitDisabled}
                data-testid={testId ? `${testId}-create-button` : undefined}
                className={`
                  w-full py-3 px-4 rounded-lg font-semibold text-base
                  transition-all duration-200
                  focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
                  ${isSubmitDisabled
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                  }
                `}
              >
                {isCreating ? '作成中...' : 'シェア画像を作成'}
              </button>
            </>
          ) : (
            <>
              {/* プライバシー警告 */}
              <div
                data-testid={testId ? `${testId}-privacy-warning` : undefined}
                className="mb-4 p-4 bg-amber-50 border border-amber-200 rounded-lg"
              >
                <ul className="space-y-1 text-sm text-amber-800 mb-3">
                  {PRIVACY_WARNINGS.map((warning, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <svg
                        className="w-4 h-4 mt-0.5 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                        aria-hidden="true"
                      >
                        <path
                          fillRule="evenodd"
                          d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                          clipRule="evenodd"
                        />
                      </svg>
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
                    className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-900">
                    上記を理解しました
                  </span>
                </label>
              </div>

              {/* シェアボタン群 */}
              <div
                data-testid={testId ? `${testId}-share-buttons` : undefined}
                className="grid grid-cols-2 gap-3"
              >
                <button
                  type="button"
                  onClick={handleTwitterShare}
                  disabled={!privacyAgreed}
                  data-testid={testId ? `${testId}-share-twitter` : undefined}
                  className={`
                    flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium
                    transition-all duration-200
                    ${privacyAgreed
                      ? 'bg-black text-white hover:bg-gray-800'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }
                  `}
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                  Xでシェア
                </button>

                <button
                  type="button"
                  onClick={handleLineShare}
                  disabled={!privacyAgreed}
                  data-testid={testId ? `${testId}-share-line` : undefined}
                  className={`
                    flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium
                    transition-all duration-200
                    ${privacyAgreed
                      ? 'bg-green-500 text-white hover:bg-green-600'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }
                  `}
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M19.365 9.863c.349 0 .63.285.63.631 0 .345-.281.63-.63.63H17.61v1.125h1.755c.349 0 .63.283.63.63 0 .344-.281.629-.63.629h-2.386a.63.63 0 0 1-.63-.629V8.108a.63.63 0 0 1 .63-.63h2.386c.349 0 .63.285.63.63 0 .349-.281.63-.63.63H17.61v1.125h1.755zm-3.855 3.016a.63.63 0 0 1-.63.63.629.629 0 0 1-.511-.26l-2.4-3.255v2.885c0 .349-.283.63-.63.63a.63.63 0 0 1-.63-.63V8.108a.63.63 0 0 1 .63-.63c.2 0 .385.098.5.26l2.415 3.254V8.108c0-.345.285-.63.63-.63s.626.285.626.63v4.771zm-5.741 0a.63.63 0 0 1-.63.63.63.63 0 0 1-.63-.63V8.108a.63.63 0 0 1 .63-.63c.349 0 .63.285.63.63v4.771zm-2.466.63H4.917a.63.63 0 0 1-.63-.63V8.108a.63.63 0 0 1 .63-.63c.349 0 .63.285.63.63v4.141h1.756c.348 0 .629.283.629.63 0 .344-.281.63-.629.63M24 10.314C24 4.943 18.615.572 12 .572S0 4.943 0 10.314c0 4.811 4.27 8.842 10.035 9.608.391.082.923.258 1.058.59.12.301.079.766.038 1.08l-.164 1.02c-.045.301-.24 1.186 1.049.645 1.291-.539 6.916-4.078 9.436-6.975C23.176 14.393 24 12.458 24 10.314" />
                  </svg>
                  LINEでシェア
                </button>

                <button
                  type="button"
                  onClick={handleInstagramShare}
                  disabled={!privacyAgreed}
                  data-testid={testId ? `${testId}-share-instagram` : undefined}
                  className={`
                    flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium
                    transition-all duration-200
                    ${privacyAgreed
                      ? 'bg-gradient-to-r from-purple-500 via-pink-500 to-orange-500 text-white hover:opacity-90'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }
                  `}
                >
                  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zM12 0C8.741 0 8.333.014 7.053.072 2.695.272.273 2.69.073 7.052.014 8.333 0 8.741 0 12c0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98C8.333 23.986 8.741 24 12 24c3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98C15.668.014 15.259 0 12 0zm0 5.838a6.162 6.162 0 100 12.324 6.162 6.162 0 000-12.324zM12 16a4 4 0 110-8 4 4 0 010 8zm6.406-11.845a1.44 1.44 0 100 2.881 1.44 1.44 0 000-2.881z" />
                  </svg>
                  Instagramでシェア
                </button>

                <button
                  type="button"
                  onClick={handleDownload}
                  disabled={!privacyAgreed}
                  data-testid={testId ? `${testId}-download` : undefined}
                  className={`
                    flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium
                    transition-all duration-200
                    ${privacyAgreed
                      ? 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    }
                  `}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  画像をダウンロード
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
