/**
 * ShareSection コンポーネントのテスト
 *
 * UC-013: シェア画像の選択
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import React from 'react'

// モックコンポーネント（実装前のテスト用）
const ShareSection = ({
  images,
  onShare,
  isAuthenticated,
  onLoginRequired,
  testId,
}: {
  images: Array<{ id: string; type: 'morph' | 'parts'; imageUrl: string; label: string }>
  onShare: (imageId: string) => void
  isAuthenticated: boolean
  onLoginRequired: () => void
  testId?: string
}) => {
  const [selectedId, setSelectedId] = React.useState<string | null>(null)

  const handleShare = () => {
    if (!isAuthenticated) {
      onLoginRequired()
      return
    }
    if (selectedId) {
      onShare(selectedId)
    }
  }

  return (
    <div data-testid={testId}>
      <h3>シェア</h3>
      <p>シェアする画像を選択してください</p>
      <div data-testid={`${testId}-images`}>
        {images.map((img) => (
          <button
            key={img.id}
            type="button"
            onClick={() => setSelectedId(img.id)}
            data-testid={`${testId}-image-${img.id}`}
            data-selected={selectedId === img.id}
          >
            <img src={img.imageUrl} alt={img.label} />
            <span>{img.label}</span>
          </button>
        ))}
      </div>
      <button
        type="button"
        onClick={handleShare}
        disabled={!selectedId}
        data-testid={`${testId}-share-button`}
      >
        この画像をシェア
      </button>
    </div>
  )
}

describe('ShareSection', () => {
  const mockImages = [
    { id: 'morph-1', type: 'morph' as const, imageUrl: 'data:image/png;base64,abc', label: '全体結果' },
    { id: 'parts-1', type: 'parts' as const, imageUrl: 'data:image/png;base64,def', label: 'パーツ別結果' },
  ]

  describe('画像選択', () => {
    it('生成済みの画像が一覧表示される', () => {
      const onShare = vi.fn()
      const onLoginRequired = vi.fn()

      render(
        <ShareSection
          images={mockImages}
          onShare={onShare}
          isAuthenticated={true}
          onLoginRequired={onLoginRequired}
          testId="share-section"
        />
      )

      expect(screen.getByTestId('share-section-image-morph-1')).toBeInTheDocument()
      expect(screen.getByTestId('share-section-image-parts-1')).toBeInTheDocument()
    })

    it('画像をクリックすると選択状態になる', () => {
      const onShare = vi.fn()
      const onLoginRequired = vi.fn()

      render(
        <ShareSection
          images={mockImages}
          onShare={onShare}
          isAuthenticated={true}
          onLoginRequired={onLoginRequired}
          testId="share-section"
        />
      )

      const imageButton = screen.getByTestId('share-section-image-morph-1')
      fireEvent.click(imageButton)

      expect(imageButton).toHaveAttribute('data-selected', 'true')
    })

    it('画像が選択されていない状態ではシェアボタンが無効', () => {
      const onShare = vi.fn()
      const onLoginRequired = vi.fn()

      render(
        <ShareSection
          images={mockImages}
          onShare={onShare}
          isAuthenticated={true}
          onLoginRequired={onLoginRequired}
          testId="share-section"
        />
      )

      const shareButton = screen.getByTestId('share-section-share-button')
      expect(shareButton).toBeDisabled()
    })

    it('画像選択後にシェアボタンが有効になる', () => {
      const onShare = vi.fn()
      const onLoginRequired = vi.fn()

      render(
        <ShareSection
          images={mockImages}
          onShare={onShare}
          isAuthenticated={true}
          onLoginRequired={onLoginRequired}
          testId="share-section"
        />
      )

      fireEvent.click(screen.getByTestId('share-section-image-morph-1'))
      const shareButton = screen.getByTestId('share-section-share-button')
      expect(shareButton).not.toBeDisabled()
    })
  })

  describe('認証状態', () => {
    it('未認証の場合、シェアボタンクリックでログイン誘導', () => {
      const onShare = vi.fn()
      const onLoginRequired = vi.fn()

      render(
        <ShareSection
          images={mockImages}
          onShare={onShare}
          isAuthenticated={false}
          onLoginRequired={onLoginRequired}
          testId="share-section"
        />
      )

      fireEvent.click(screen.getByTestId('share-section-image-morph-1'))
      fireEvent.click(screen.getByTestId('share-section-share-button'))

      expect(onLoginRequired).toHaveBeenCalled()
      expect(onShare).not.toHaveBeenCalled()
    })

    it('認証済みの場合、onShareが呼ばれる', () => {
      const onShare = vi.fn()
      const onLoginRequired = vi.fn()

      render(
        <ShareSection
          images={mockImages}
          onShare={onShare}
          isAuthenticated={true}
          onLoginRequired={onLoginRequired}
          testId="share-section"
        />
      )

      fireEvent.click(screen.getByTestId('share-section-image-morph-1'))
      fireEvent.click(screen.getByTestId('share-section-share-button'))

      expect(onShare).toHaveBeenCalledWith('morph-1')
      expect(onLoginRequired).not.toHaveBeenCalled()
    })
  })
})
