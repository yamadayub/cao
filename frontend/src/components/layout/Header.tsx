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
        className="text-gray-600 hover:text-gray-900 transition-colors"
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
            className="text-gray-600 hover:text-gray-900 transition-colors"
          >
            マイページ
          </Link>
          <UserButton afterSignOutUrl="/" />
        </>
      ) : (
        <SignInButton mode="modal">
          <button className="text-gray-600 hover:text-gray-900 transition-colors">
            ログイン
          </button>
        </SignInButton>
      )}
    </>
  )
}

export function Header({ variant = 'default' }: HeaderProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const headerClasses =
    variant === 'transparent'
      ? 'absolute top-0 left-0 right-0 z-10'
      : 'bg-white shadow-sm'

  return (
    <header className={headerClasses}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="text-2xl font-bold text-blue-600">
            Cao
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-4">
            {mounted && <HeaderAuthSection />}
            <Link
              href="/simulate"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              今すぐ試す
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}
