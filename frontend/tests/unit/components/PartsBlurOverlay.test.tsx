/**
 * PartsBlurOverlayコンポーネント - 単体テスト
 *
 * 対象: パーツ別シミュレーション結果のブラー表示コンポーネント
 * 参照: functional-spec.md セクション 3.4 (SCR-003) パーツ別適用モード
 *
 * 要件:
 * 1. 未認証ユーザーがパーツ別シミュレーションを実行した場合、結果画像にブラーが適用される
 * 2. 認証済みユーザーがパーツ別シミュレーションを実行した場合、結果画像はブラーなしで表示される
 * 3. 未認証ユーザーがブラー画像をタップすると、ログイン誘導モーダルが表示される
 * 4. ログイン後、パーツ別シミュレーション結果がブラーなしで表示される
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { PartsBlurOverlay, type PartsBlurOverlayProps } from '@/components/features/PartsBlurOverlay'

// useAuthSafeフックのモック
const mockUseAuthSafe = vi.fn()
vi.mock('@/hooks/useClerkSafe', () => ({
  useAuthSafe: () => mockUseAuthSafe(),
  useUserSafe: () => mockUseAuthSafe(),
  isClerkAvailable: () => true,
}))

describe('PartsBlurOverlay', () => {
  const defaultProps: PartsBlurOverlayProps = {
    imageUrl: 'data:image/png;base64,test-image-data',
    altText: 'パーツ別シミュレーション結果',
    onLoginClick: vi.fn(),
    testId: 'parts-blur-overlay',
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('未認証ユーザーの場合', () => {
    beforeEach(() => {
      // 未認証状態をモック
      mockUseAuthSafe.mockReturnValue({
        isSignedIn: false,
        user: null,
        userId: null,
        isLoaded: true,
      })
    })

    it('結果画像にブラーが適用される', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={false} />)

      const blurOverlay = screen.getByTestId('parts-blur-overlay-blur')
      expect(blurOverlay).toBeInTheDocument()
      expect(blurOverlay).toHaveClass('blur')
    })

    it('画像は表示されるがブラー効果がかかっている', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={false} />)

      const image = screen.getByAltText('パーツ別シミュレーション結果')
      expect(image).toBeInTheDocument()
      expect(image).toHaveAttribute('src', 'data:image/png;base64,test-image-data')

      // 画像コンテナにブラークラスが適用されている
      const container = screen.getByTestId('parts-blur-overlay-image-container')
      expect(container).toHaveClass('blur-lg')
    })

    it('「タップしてログイン」のテキストが表示される', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={false} />)

      expect(screen.getByText('partsBlur.tapToLogin')).toBeInTheDocument()
    })

    it('ブラー画像をタップするとonLoginClickが呼ばれる', () => {
      const onLoginClick = vi.fn()
      render(
        <PartsBlurOverlay
          {...defaultProps}
          isAuthenticated={false}
          onLoginClick={onLoginClick}
        />
      )

      const overlay = screen.getByTestId('parts-blur-overlay')
      fireEvent.click(overlay)

      expect(onLoginClick).toHaveBeenCalledTimes(1)
    })

    it('Enterキーでもログイン誘導が発火する（アクセシビリティ）', () => {
      const onLoginClick = vi.fn()
      render(
        <PartsBlurOverlay
          {...defaultProps}
          isAuthenticated={false}
          onLoginClick={onLoginClick}
        />
      )

      const overlay = screen.getByTestId('parts-blur-overlay')
      fireEvent.keyDown(overlay, { key: 'Enter' })

      expect(onLoginClick).toHaveBeenCalledTimes(1)
    })

    it('スペースキーでもログイン誘導が発火する（アクセシビリティ）', () => {
      const onLoginClick = vi.fn()
      render(
        <PartsBlurOverlay
          {...defaultProps}
          isAuthenticated={false}
          onLoginClick={onLoginClick}
        />
      )

      const overlay = screen.getByTestId('parts-blur-overlay')
      fireEvent.keyDown(overlay, { key: ' ' })

      expect(onLoginClick).toHaveBeenCalledTimes(1)
    })

    it('aria-label属性が設定されている', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={false} />)

      const overlay = screen.getByTestId('parts-blur-overlay')
      expect(overlay).toHaveAttribute('aria-label', 'partsBlur.loginAriaLabel')
    })

    it('role="button"が設定されている（クリック可能なことを示す）', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={false} />)

      const overlay = screen.getByTestId('parts-blur-overlay')
      expect(overlay).toHaveAttribute('role', 'button')
    })

    it('tabIndexが設定されキーボードフォーカス可能', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={false} />)

      const overlay = screen.getByTestId('parts-blur-overlay')
      expect(overlay).toHaveAttribute('tabIndex', '0')
    })
  })

  describe('認証済みユーザーの場合', () => {
    beforeEach(() => {
      // 認証済み状態をモック
      mockUseAuthSafe.mockReturnValue({
        isSignedIn: true,
        user: { id: 'user-123', primaryEmailAddress: { emailAddress: 'test@example.com' } },
        userId: 'user-123',
        isLoaded: true,
      })
    })

    it('結果画像にブラーが適用されない', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={true} />)

      const blurOverlay = screen.queryByTestId('parts-blur-overlay-blur')
      expect(blurOverlay).not.toBeInTheDocument()
    })

    it('画像がクリアに表示される', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={true} />)

      const image = screen.getByAltText('パーツ別シミュレーション結果')
      expect(image).toBeInTheDocument()
      expect(image).toHaveAttribute('src', 'data:image/png;base64,test-image-data')

      // ブラークラスが適用されていない
      const container = screen.getByTestId('parts-blur-overlay-image-container')
      expect(container).not.toHaveClass('blur-lg')
    })

    it('「タップしてログイン」のテキストが表示されない', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={true} />)

      expect(screen.queryByText('タップしてログイン')).not.toBeInTheDocument()
    })

    it('画像クリックしてもonLoginClickは呼ばれない', () => {
      const onLoginClick = vi.fn()
      render(
        <PartsBlurOverlay
          {...defaultProps}
          isAuthenticated={true}
          onLoginClick={onLoginClick}
        />
      )

      const container = screen.getByTestId('parts-blur-overlay')
      fireEvent.click(container)

      expect(onLoginClick).not.toHaveBeenCalled()
    })

    it('role属性がimgである（単なる画像表示）', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={true} />)

      const overlay = screen.getByTestId('parts-blur-overlay')
      expect(overlay).not.toHaveAttribute('role', 'button')
    })

    it('tabIndexが設定されていない（フォーカス対象外）', () => {
      render(<PartsBlurOverlay {...defaultProps} isAuthenticated={true} />)

      const overlay = screen.getByTestId('parts-blur-overlay')
      expect(overlay).not.toHaveAttribute('tabIndex', '0')
    })
  })

  describe('ローディング状態', () => {
    it('ローディング中はスケルトン表示される', () => {
      render(<PartsBlurOverlay {...defaultProps} isLoading={true} />)

      const skeleton = screen.getByTestId('parts-blur-overlay-skeleton')
      expect(skeleton).toBeInTheDocument()
    })

    it('ローディング中は画像が表示されない', () => {
      render(<PartsBlurOverlay {...defaultProps} isLoading={true} />)

      const image = screen.queryByAltText('パーツ別シミュレーション結果')
      expect(image).not.toBeInTheDocument()
    })
  })

  describe('エラー状態', () => {
    it('画像URLがない場合はプレースホルダーが表示される', () => {
      render(<PartsBlurOverlay {...defaultProps} imageUrl="" />)

      const placeholder = screen.getByTestId('parts-blur-overlay-placeholder')
      expect(placeholder).toBeInTheDocument()
    })

    it('画像URLがnullの場合はプレースホルダーが表示される', () => {
      render(<PartsBlurOverlay {...defaultProps} imageUrl={null as unknown as string} />)

      const placeholder = screen.getByTestId('parts-blur-overlay-placeholder')
      expect(placeholder).toBeInTheDocument()
    })
  })

  describe('画像サイズ', () => {
    it('デフォルトサイズで表示される', () => {
      render(<PartsBlurOverlay {...defaultProps} />)

      const container = screen.getByTestId('parts-blur-overlay')
      expect(container).toBeInTheDocument()
    })

    it('カスタムサイズを指定できる', () => {
      render(
        <PartsBlurOverlay
          {...defaultProps}
          width={300}
          height={300}
        />
      )

      const container = screen.getByTestId('parts-blur-overlay')
      expect(container).toHaveStyle({ width: '300px', height: '300px' })
    })
  })

  describe('ログイン誘導モーダルとの連携', () => {
    beforeEach(() => {
      mockUseAuthSafe.mockReturnValue({
        isSignedIn: false,
        user: null,
        userId: null,
        isLoaded: true,
      })
    })

    it('ブラー画像タップ後にモーダルが開かれることを想定したコールバックが実行される', () => {
      const onLoginClick = vi.fn()
      render(
        <PartsBlurOverlay
          {...defaultProps}
          isAuthenticated={false}
          onLoginClick={onLoginClick}
        />
      )

      const overlay = screen.getByTestId('parts-blur-overlay')
      fireEvent.click(overlay)

      // onLoginClickが呼ばれ、親コンポーネントでモーダルを開く処理が期待される
      expect(onLoginClick).toHaveBeenCalledTimes(1)
    })
  })

  describe('ログイン後の状態変更', () => {
    it('isAuthenticated が false から true に変わるとブラーが解除される', async () => {
      const { rerender } = render(
        <PartsBlurOverlay {...defaultProps} isAuthenticated={false} />
      )

      // 初期状態: ブラーあり
      expect(screen.getByTestId('parts-blur-overlay-blur')).toBeInTheDocument()

      // ログイン後: isAuthenticated を true に変更
      rerender(<PartsBlurOverlay {...defaultProps} isAuthenticated={true} />)

      // ブラーが解除される
      await waitFor(() => {
        expect(screen.queryByTestId('parts-blur-overlay-blur')).not.toBeInTheDocument()
      })
    })

    it('認証状態が変わると画像は同じものが表示され続ける', async () => {
      const { rerender } = render(
        <PartsBlurOverlay {...defaultProps} isAuthenticated={false} />
      )

      const imageBefore = screen.getByAltText('パーツ別シミュレーション結果')
      expect(imageBefore).toHaveAttribute('src', 'data:image/png;base64,test-image-data')

      // ログイン後
      rerender(<PartsBlurOverlay {...defaultProps} isAuthenticated={true} />)

      await waitFor(() => {
        const imageAfter = screen.getByAltText('パーツ別シミュレーション結果')
        expect(imageAfter).toHaveAttribute('src', 'data:image/png;base64,test-image-data')
      })
    })
  })

  describe('パーツ別シミュレーション特有のUI', () => {
    beforeEach(() => {
      mockUseAuthSafe.mockReturnValue({
        isSignedIn: false,
        user: null,
        userId: null,
        isLoaded: true,
      })
    })

    it('パーツ別の結果であることを示すラベルが表示される', () => {
      render(
        <PartsBlurOverlay
          {...defaultProps}
          isAuthenticated={false}
          showPartsLabel={true}
          appliedParts={['left_eye', 'right_eye', 'nose']}
        />
      )

      // With mocked translations, keys are returned directly
      expect(screen.getByText(/parts\.appliedParts/)).toBeInTheDocument()
    })

    it('appliedPartsがない場合はラベルが表示されない', () => {
      render(
        <PartsBlurOverlay
          {...defaultProps}
          isAuthenticated={false}
          showPartsLabel={true}
          appliedParts={[]}
        />
      )

      expect(screen.queryByText(/parts\.appliedParts/)).not.toBeInTheDocument()
    })
  })
})

describe('PartsBlurOverlay - ログイン誘導モーダル統合テスト', () => {
  /**
   * このテストグループはPartsBlurOverlayとLoginPromptModalの連携をテスト
   * 実際のモーダル表示は親コンポーネントで制御される想定
   */

  const defaultProps: PartsBlurOverlayProps = {
    imageUrl: 'data:image/png;base64,test-image-data',
    altText: 'パーツ別シミュレーション結果',
    onLoginClick: vi.fn(),
    testId: 'parts-blur-overlay',
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockUseAuthSafe.mockReturnValue({
      isSignedIn: false,
      user: null,
      userId: null,
      isLoaded: true,
    })
  })

  it('未認証ユーザーがブラー画像をタップした際のコールバックが正しく動作する', () => {
    const mockOnLoginClick = vi.fn()

    render(
      <PartsBlurOverlay
        {...defaultProps}
        isAuthenticated={false}
        onLoginClick={mockOnLoginClick}
      />
    )

    const overlay = screen.getByTestId('parts-blur-overlay')
    fireEvent.click(overlay)

    expect(mockOnLoginClick).toHaveBeenCalledTimes(1)
  })

  it('モーダルタイトル用のプロパティが渡される', () => {
    const mockOnLoginClick = vi.fn()

    render(
      <PartsBlurOverlay
        {...defaultProps}
        isAuthenticated={false}
        onLoginClick={mockOnLoginClick}
        loginPromptTitle="パーツ別の結果を見るにはログインが必要です"
        loginPromptDescription="パーツ別シミュレーションの詳細な結果を確認するにはログインしてください。"
      />
    )

    const overlay = screen.getByTestId('parts-blur-overlay')
    fireEvent.click(overlay)

    // コールバック呼び出し時にタイトルと説明が渡されることを期待
    expect(mockOnLoginClick).toHaveBeenCalledWith({
      title: 'パーツ別の結果を見るにはログインが必要です',
      description: 'パーツ別シミュレーションの詳細な結果を確認するにはログインしてください。',
    })
  })
})
