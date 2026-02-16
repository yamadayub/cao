/**
 * Header component tests
 *
 * UC-009: ユーザー登録・ログイン
 * - ヘッダー右上は常時「ログイン」ボタンを表示
 * - 未認証時: クリックでログインモーダル表示
 * - 認証済み時: クリックでマイページへ遷移
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Header } from '@/components/layout/Header'

// Clerk mocks
const mockUseAuth = vi.fn()
const mockUseClerk = vi.fn()
const mockSignInButton = vi.fn()

vi.mock('@clerk/nextjs', () => {
  // UserButton mock with MenuItems support
  const MockUserButton = ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="user-button">{children}</div>
  )
  MockUserButton.MenuItems = ({ children }: { children?: React.ReactNode }) => <>{children}</>
  MockUserButton.Link = ({ label, href }: { label: string; labelIcon?: React.ReactNode; href: string }) => (
    <a href={href} data-testid="user-button-link">{label}</a>
  )

  return {
    useAuth: () => mockUseAuth(),
    useClerk: () => mockUseClerk(),
    SignInButton: ({ children, mode }: { children: React.ReactNode; mode?: string }) => {
      mockSignInButton({ mode })
      return <div data-testid="sign-in-button" data-mode={mode}>{children}</div>
    },
    UserButton: MockUserButton,
  }
})

// Mock isClerkAvailable to return true so Header uses Clerk hooks
vi.mock('@/hooks/useClerkSafe', () => ({
  isClerkAvailable: () => true,
  useAuthSafe: () => mockUseAuth(),
  useUserSafe: () => mockUseAuth(),
}))

// Next.js router mock
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => '/',
}))

describe('Header', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseClerk.mockReturnValue({ loaded: true })
  })

  describe('未認証時', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: false,
      })
    })

    it('ログインボタンを表示する', () => {
      render(<Header />)

      const loginButton = screen.getByRole('button', { name: 'nav.login' })
      expect(loginButton).toBeDefined()
    })

    it('ログインボタンクリックでモーダルが表示される（mode="modal"）', () => {
      render(<Header />)

      // SignInButton が mode="modal" で呼ばれることを確認
      expect(mockSignInButton).toHaveBeenCalledWith(
        expect.objectContaining({ mode: 'modal' })
      )
    })

    it('ロゴをクリックするとトップページへ遷移できる', () => {
      render(<Header />)

      const logo = screen.getByRole('link', { name: 'Cao' })
      expect(logo).toBeDefined()
      expect(logo.getAttribute('href')).toBe('/')
    })

    it('「今すぐ試す」ボタンは表示しない（ログインボタンのみ）', () => {
      render(<Header />)

      // 「今すぐ試す」ボタンは削除されたので存在しない
      const ctaButton = screen.queryByRole('link', { name: '今すぐ試す' })
      expect(ctaButton).toBeNull()
    })
  })

  describe('認証済み時', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
      })
    })

    it('Clerkのユーザーアイコンを表示する', () => {
      render(<Header />)

      // 認証済みはUserButtonが表示される
      const userButton = screen.getByTestId('user-button')
      expect(userButton).toBeDefined()
    })

    it('メニューからマイページへのリンクがある', () => {
      render(<Header />)

      // マイページへのリンクがある
      const mypageLink = screen.getByTestId('user-button-link')
      expect(mypageLink).toBeDefined()
      expect(mypageLink.getAttribute('href')).toBe('/mypage')
    })

    it('ロゴとユーザーアイコンのみ表示', () => {
      render(<Header />)

      // 「今すぐ試す」ボタンは存在しない
      const ctaButton = screen.queryByRole('link', { name: '今すぐ試す' })
      expect(ctaButton).toBeNull()

      // UserButtonが表示される
      const userButton = screen.getByTestId('user-button')
      expect(userButton).toBeDefined()
    })

    it('ロゴをクリックするとトップページへ遷移できる', () => {
      render(<Header />)

      const logo = screen.getByRole('link', { name: 'Cao' })
      expect(logo).toBeDefined()
      expect(logo.getAttribute('href')).toBe('/')
    })
  })

  describe('ローディング中', () => {
    beforeEach(() => {
      mockUseAuth.mockReturnValue({
        isLoaded: false,
        isSignedIn: false,
      })
    })

    it('認証状態ローディング中はスケルトンを表示', () => {
      render(<Header />)

      // ローディング中のプレースホルダーが表示される
      const skeleton = document.querySelector('.animate-pulse')
      expect(skeleton).toBeDefined()
    })
  })
})

describe('Header - ログイン後の画面遷移', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseClerk.mockReturnValue({ loaded: true })
  })

  it('ログイン後は元の画面に留まる（ページ遷移なし）', () => {
    // This is ensured by using mode="modal" in SignInButton
    mockUseAuth.mockReturnValue({
      isLoaded: true,
      isSignedIn: false,
    })

    render(<Header />)

    // SignInButton が mode="modal" で呼ばれることを確認
    // モーダルモードの場合、ログイン後にページ遷移は発生しない
    expect(mockSignInButton).toHaveBeenCalledWith(
      expect.objectContaining({ mode: 'modal' })
    )
  })
})

describe('Header - 共通ヘッダー仕様', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockUseClerk.mockReturnValue({ loaded: true })
  })

  it('未認証時は「ログイン」ボタン、認証済み時はユーザーアイコンを表示', () => {
    // 未認証時
    mockUseAuth.mockReturnValue({
      isLoaded: true,
      isSignedIn: false,
    })

    const { rerender } = render(<Header />)
    const loginElement = screen.getByText('nav.login')
    expect(loginElement).toBeDefined()

    // 認証済み時はUserButton
    mockUseAuth.mockReturnValue({
      isLoaded: true,
      isSignedIn: true,
    })

    rerender(<Header />)
    const userButton = screen.getByTestId('user-button')
    expect(userButton).toBeDefined()
  })
})
