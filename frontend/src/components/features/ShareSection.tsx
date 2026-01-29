'use client'

import { useState, useCallback } from 'react'

export interface ShareImage {
  id: string
  type: 'morph' | 'parts'
  imageUrl: string
  label: string
}

export interface ShareSectionProps {
  /** シェア可能な画像リスト */
  images: ShareImage[]
  /** シェアボタンクリック時のコールバック */
  onShare: (imageId: string) => void
  /** 認証状態 */
  isAuthenticated: boolean
  /** ログイン必要時のコールバック */
  onLoginRequired: () => void
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * シェア画像選択セクション
 *
 * UC-013: シェア画像の選択
 */
export function ShareSection({
  images,
  onShare,
  isAuthenticated,
  onLoginRequired,
  testId,
}: ShareSectionProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const handleShare = useCallback(() => {
    if (!isAuthenticated) {
      onLoginRequired()
      return
    }
    if (selectedId) {
      onShare(selectedId)
    }
  }, [isAuthenticated, onLoginRequired, onShare, selectedId])

  const handleImageSelect = useCallback((id: string) => {
    setSelectedId(id)
  }, [])

  return (
    <div data-testid={testId} className="p-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        シェア
      </h3>
      <p className="text-sm text-gray-600 mb-4">
        シェアする画像を選択してください
      </p>

      {/* 画像選択グリッド */}
      <div
        data-testid={testId ? `${testId}-images` : undefined}
        className="grid grid-cols-2 gap-3 mb-4"
      >
        {images.map((img) => (
          <button
            key={img.id}
            type="button"
            onClick={() => handleImageSelect(img.id)}
            data-testid={testId ? `${testId}-image-${img.id}` : undefined}
            data-selected={selectedId === img.id}
            className={`
              relative rounded-lg overflow-hidden border-2 transition-all duration-200
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
              ${selectedId === img.id
                ? 'border-blue-600 ring-2 ring-blue-600 ring-opacity-50'
                : 'border-gray-200 hover:border-gray-300'
              }
            `}
          >
            <img
              src={img.imageUrl}
              alt={img.label}
              className="w-full aspect-square object-cover"
            />
            <span className={`
              absolute bottom-0 left-0 right-0 py-1 px-2 text-xs text-center
              ${selectedId === img.id
                ? 'bg-blue-600 text-white'
                : 'bg-black/50 text-white'
              }
            `}>
              {img.label}
            </span>
            {/* 選択マーク */}
            {selectedId === img.id && (
              <div className="absolute top-2 right-2 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                <svg
                  className="w-4 h-4 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={3}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* シェアボタン */}
      <button
        type="button"
        onClick={handleShare}
        disabled={!selectedId}
        data-testid={testId ? `${testId}-share-button` : undefined}
        className={`
          w-full py-3 px-4 rounded-lg font-semibold text-base
          transition-all duration-200
          focus:outline-none focus:ring-2 focus:ring-offset-2
          ${selectedId
            ? 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500'
            : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          }
        `}
      >
        この画像をシェア
      </button>
    </div>
  )
}
