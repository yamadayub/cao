'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

interface HeaderProps {
  variant?: 'default' | 'transparent'
}

function HeaderAuthSection() {
  const [isReady, setIsReady] = useState(false)
  const [isSignedIn, setIsSignedIn] = useState(false)
  const [ClerkComponents, setClerkComponents] = useState<{
    UserButton: React.ComponentType<{ afterSignOutUrl: string }>
    SignInButton: React.ComponentType<{ mode: string; children: React.ReactNode }>
  } | null>(null)

  useEffect(() => {
    // ClerkProviderが利用可能かチェック
    const checkClerk = async () => {
      try {
        const clerk = await import('@clerk/nextjs')

        // useAuthはhookなので、ここでは直接使えない
        // 代わりにClerkクライアントを使用
        const win = window as unknown as { Clerk?: { user?: unknown } }
        if (win.Clerk) {
          setIsSignedIn(!!win.Clerk.user)
          setClerkComponents({
            UserButton: clerk.UserButton,
            SignInButton: clerk.SignInButton as React.ComponentType<{ mode: string; children: React.ReactNode }>,
          })
          setIsReady(true)
        } else {
          // Clerkがまだロードされていない場合は少し待って再試行
          setTimeout(() => {
            const win2 = window as unknown as { Clerk?: { user?: unknown } }
            if (win2.Clerk) {
              setIsSignedIn(!!win2.Clerk.user)
              setClerkComponents({
                UserButton: clerk.UserButton,
                SignInButton: clerk.SignInButton as React.ComponentType<{ mode: string; children: React.ReactNode }>,
              })
              setIsReady(true)
            }
          }, 500)
        }
      } catch {
        // Clerkが利用できない場合は何も表示しない
        console.warn('Clerk is not available')
      }
    }

    checkClerk()
  }, [])

  if (!isReady || !ClerkComponents) {
    // Clerkがまだ準備できていない場合はログインボタンを表示
    return (
      <Link
        href="/sign-in"
        className="text-sm text-neutral-600 hover:text-primary-700 transition-colors duration-300"
      >
        ログイン
      </Link>
    )
  }

  const { UserButton, SignInButton } = ClerkComponents

  return (
    <>
      {isSignedIn ? (
        <>
          <Link
            href="/mypage"
            className="text-sm text-neutral-600 hover:text-primary-700 transition-colors duration-300"
          >
            マイページ
          </Link>
          <UserButton afterSignOutUrl="/" />
        </>
      ) : (
        <SignInButton mode="modal">
          <button className="text-sm text-neutral-600 hover:text-primary-700 transition-colors duration-300">
            ログイン
          </button>
        </SignInButton>
      )}
    </>
  )
}

export function Header({ variant = 'default' }: HeaderProps) {
  const [mounted, setMounted] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    setMounted(true)

    const handleScroll = () => {
      setScrolled(window.scrollY > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const headerClasses =
    variant === 'transparent'
      ? `fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrolled ? 'bg-white/95 backdrop-blur-sm shadow-elegant' : 'bg-transparent'
        }`
      : 'bg-white border-b border-neutral-100'

  return (
    <header className={headerClasses}>
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link
            href="/"
            className="font-serif text-3xl font-medium text-primary-700 tracking-tight hover:text-primary-800 transition-colors duration-300"
          >
            Cao
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-8">
            {mounted && <HeaderAuthSection />}
            <Link
              href="/simulate"
              className="btn-primary text-sm px-6 py-3"
            >
              今すぐ試す
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}
