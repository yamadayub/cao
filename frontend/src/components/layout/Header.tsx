'use client'

import Link from 'next/link'
import { useAuth, UserButton, SignInButton } from '@clerk/nextjs'

interface HeaderProps {
  variant?: 'default' | 'transparent'
}

export function Header({ variant = 'default' }: HeaderProps) {
  const { isSignedIn, isLoaded } = useAuth()

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
            {isLoaded && (
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
            )}
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
