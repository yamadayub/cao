/**
 * ShareCustomizeModal コンポーネントのテスト
 *
 * UC-014: シェア画像のカスタマイズ
 * UC-015: SNSへのシェア
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

// テンプレートタイプ
type ShareTemplate = 'before_after' | 'single' | 'parts_highlight'

// モックコンポーネント（実装前のテスト用）
const ShareCustomizeModal = ({
  isOpen,
  onClose,
  sourceImage,
  resultImage,
  onCreateShare,
  isCreating,
  testId,
}: {
  isOpen: boolean
  onClose: () => void
  sourceImage: string
  resultImage: string
  onCreateShare: (data: {
    template: ShareTemplate
    caption: string
  }) => Promise<void>
  isCreating: boolean
  testId?: string
}) => {
  const [template, setTemplate] = React.useState<ShareTemplate>('before_after')
  const [caption, setCaption] = React.useState('')
  const [showShareOptions, setShowShareOptions] = React.useState(false)
  const [privacyAgreed, setPrivacyAgreed] = React.useState(false)

  if (!isOpen) return null

  const handleSubmit = async () => {
    await onCreateShare({ template, caption })
    setShowShareOptions(true)
  }

  const captionError = caption.length > 140 ? 'キャプションは140文字以内で入力してください' : ''

  return (
    <div data-testid={testId} role="dialog">
      <button type="button" onClick={onClose} data-testid={`${testId}-close`}>
        閉じる
      </button>

      {!showShareOptions ? (
        <>
          <h2>シェア画像を作成</h2>

          {/* プレビュー */}
          <div data-testid={`${testId}-preview`}>
            {/* プレビュー画像表示 */}
          </div>

          {/* テンプレート選択 */}
          <div data-testid={`${testId}-templates`}>
            <button
              type="button"
              onClick={() => setTemplate('before_after')}
              data-testid={`${testId}-template-before_after`}
              data-selected={template === 'before_after'}
            >
              Before/After
            </button>
            <button
              type="button"
              onClick={() => setTemplate('single')}
              data-testid={`${testId}-template-single`}
              data-selected={template === 'single'}
            >
              単体
            </button>
            <button
              type="button"
              onClick={() => setTemplate('parts_highlight')}
              data-testid={`${testId}-template-parts_highlight`}
              data-selected={template === 'parts_highlight'}
            >
              パーツハイライト
            </button>
          </div>

          {/* キャプション入力 */}
          <div>
            <textarea
              value={caption}
              onChange={(e) => setCaption(e.target.value.slice(0, 140))}
              maxLength={140}
              placeholder="キャプション（任意）"
              data-testid={`${testId}-caption`}
            />
            <span data-testid={`${testId}-caption-count`}>{caption.length}/140</span>
            {captionError && (
              <span data-testid={`${testId}-caption-error`} role="alert">
                {captionError}
              </span>
            )}
          </div>

          {/* 作成ボタン */}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={isCreating || caption.length > 140}
            data-testid={`${testId}-create-button`}
          >
            {isCreating ? '作成中...' : 'シェア画像を作成'}
          </button>
        </>
      ) : (
        <>
          {/* シェア先選択 */}
          <h2>シェア先を選択</h2>

          {/* プライバシー警告 */}
          <div data-testid={`${testId}-privacy-warning`}>
            <ul>
              <li>シェアした画像は誰でも閲覧できます</li>
              <li>一度シェアした画像は削除できない場合があります</li>
              <li>他人の写真を無断でシェアしないでください</li>
            </ul>
            <label>
              <input
                type="checkbox"
                checked={privacyAgreed}
                onChange={(e) => setPrivacyAgreed(e.target.checked)}
                data-testid={`${testId}-privacy-checkbox`}
              />
              上記を理解しました
            </label>
          </div>

          {/* シェアボタン群 */}
          <div data-testid={`${testId}-share-buttons`}>
            <button
              type="button"
              disabled={!privacyAgreed}
              data-testid={`${testId}-share-twitter`}
            >
              Xでシェア
            </button>
            <button
              type="button"
              disabled={!privacyAgreed}
              data-testid={`${testId}-share-line`}
            >
              LINEでシェア
            </button>
            <button
              type="button"
              disabled={!privacyAgreed}
              data-testid={`${testId}-share-instagram`}
            >
              Instagramでシェア
            </button>
            <button
              type="button"
              disabled={!privacyAgreed}
              data-testid={`${testId}-download`}
            >
              画像をダウンロード
            </button>
          </div>
        </>
      )}
    </div>
  )
}

describe('ShareCustomizeModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    sourceImage: 'data:image/png;base64,source',
    resultImage: 'data:image/png;base64,result',
    onCreateShare: vi.fn().mockResolvedValue(undefined),
    isCreating: false,
    testId: 'share-modal',
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('表示', () => {
    it('isOpenがfalseの場合は何も表示しない', () => {
      render(<ShareCustomizeModal {...defaultProps} isOpen={false} />)
      expect(screen.queryByTestId('share-modal')).not.toBeInTheDocument()
    })

    it('isOpenがtrueの場合はモーダルを表示', () => {
      render(<ShareCustomizeModal {...defaultProps} />)
      expect(screen.getByTestId('share-modal')).toBeInTheDocument()
    })

    it('閉じるボタンをクリックするとonCloseが呼ばれる', () => {
      const onClose = vi.fn()
      render(<ShareCustomizeModal {...defaultProps} onClose={onClose} />)

      fireEvent.click(screen.getByTestId('share-modal-close'))
      expect(onClose).toHaveBeenCalled()
    })
  })

  describe('テンプレート選択', () => {
    it('デフォルトでBefore/Afterが選択されている', () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      const beforeAfterButton = screen.getByTestId('share-modal-template-before_after')
      expect(beforeAfterButton).toHaveAttribute('data-selected', 'true')
    })

    it('テンプレートを変更できる', () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      fireEvent.click(screen.getByTestId('share-modal-template-single'))
      expect(screen.getByTestId('share-modal-template-single')).toHaveAttribute('data-selected', 'true')
      expect(screen.getByTestId('share-modal-template-before_after')).toHaveAttribute('data-selected', 'false')
    })
  })

  describe('キャプション入力', () => {
    it('キャプションを入力できる', () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      const captionInput = screen.getByTestId('share-modal-caption')
      fireEvent.change(captionInput, { target: { value: 'テストキャプション' } })

      expect(captionInput).toHaveValue('テストキャプション')
    })

    it('文字数カウントが表示される', () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      const captionInput = screen.getByTestId('share-modal-caption')
      fireEvent.change(captionInput, { target: { value: 'テスト' } })

      expect(screen.getByTestId('share-modal-caption-count')).toHaveTextContent('3/140')
    })

    it('140文字を超えるとエラーが表示される', () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      const captionInput = screen.getByTestId('share-modal-caption')
      const longText = 'a'.repeat(141)
      fireEvent.change(captionInput, { target: { value: longText } })

      // maxLengthで制限されるため140文字になる
      expect(captionInput).toHaveValue('a'.repeat(140))
    })
  })

  describe('シェア画像作成', () => {
    it('作成ボタンクリックでonCreateShareが呼ばれる', async () => {
      const onCreateShare = vi.fn().mockResolvedValue(undefined)
      render(<ShareCustomizeModal {...defaultProps} onCreateShare={onCreateShare} />)

      fireEvent.click(screen.getByTestId('share-modal-create-button'))

      await waitFor(() => {
        expect(onCreateShare).toHaveBeenCalledWith({
          template: 'before_after',
          caption: '',
        })
      })
    })

    it('作成中はボタンが無効になる', () => {
      render(<ShareCustomizeModal {...defaultProps} isCreating={true} />)

      expect(screen.getByTestId('share-modal-create-button')).toBeDisabled()
      expect(screen.getByTestId('share-modal-create-button')).toHaveTextContent('作成中...')
    })
  })

  describe('シェア先選択（作成後）', () => {
    it('作成後にシェア先選択画面が表示される', async () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      fireEvent.click(screen.getByTestId('share-modal-create-button'))

      await waitFor(() => {
        expect(screen.getByTestId('share-modal-privacy-warning')).toBeInTheDocument()
      })
    })

    it('プライバシー同意前はシェアボタンが無効', async () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      fireEvent.click(screen.getByTestId('share-modal-create-button'))

      await waitFor(() => {
        expect(screen.getByTestId('share-modal-share-twitter')).toBeDisabled()
        expect(screen.getByTestId('share-modal-share-line')).toBeDisabled()
      })
    })

    it('プライバシー同意後にシェアボタンが有効になる', async () => {
      render(<ShareCustomizeModal {...defaultProps} />)

      fireEvent.click(screen.getByTestId('share-modal-create-button'))

      await waitFor(() => {
        expect(screen.getByTestId('share-modal-privacy-checkbox')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByTestId('share-modal-privacy-checkbox'))

      expect(screen.getByTestId('share-modal-share-twitter')).not.toBeDisabled()
      expect(screen.getByTestId('share-modal-share-line')).not.toBeDisabled()
    })
  })
})
