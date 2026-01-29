'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useAuth, useClerk, UserButton, SignInButton } from '@clerk/nextjs'

interface HeaderProps {
  variant?: 'default' | 'transparent'
}

interface HeaderAuthSectionProps {
  onAuthStateChange?: (isSignedIn: boolean) => void
}

function HeaderAuthSection({ onAuthStateChange }: HeaderAuthSectionProps) {
  const { isLoaded, isSignedIn } = useAuth()
  const { loaded: clerkLoaded } = useClerk()

  useEffect(() => {
    if (isLoaded) {
      onAuthStateChange?.(!!isSignedIn)
    }
  }, [isLoaded, isSignedIn, onAuthStateChange])

  // Clerkがまだロードされていない場合
  if (!isLoaded || !clerkLoaded) {
    return (
      <div className="w-8 h-8 rounded-full bg-neutral-100 animate-pulse" />
    )
  }

  return (
    <>
      {isSignedIn ? (
        <>
          <Link
            href="/simulate"
            className="hidden md:inline text-sm text-neutral-600 hover:text-primary-700 transition-colors duration-300"
          >
            シミュレーション
          </Link>
          <Link
            href="/mypage"
            className="hidden md:inline text-sm text-neutral-600 hover:text-primary-700 transition-colors duration-300"
          >
            マイページ
          </Link>
          <UserButton afterSignOutUrl="/" />
        </>
      ) : (
        <SignInButton mode="modal">
          <button className="hidden md:inline text-sm text-neutral-600 hover:text-primary-700 transition-colors duration-300">
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
  const [isSignedIn, setIsSignedIn] = useState(false)

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
      <div className="max-w-6xl mx-auto px-4 md:px-6">
        <div className="flex items-center justify-between h-16 md:h-20">
          {/* Logo */}
          <Link
            href="/"
            className="font-serif text-2xl md:text-3xl font-medium text-primary-700 tracking-tight hover:text-primary-800 transition-colors duration-300"
          >
            Cao
          </Link>

          {/* Navigation */}
          <nav className="flex items-center gap-3 md:gap-8">
            {mounted && <HeaderAuthSection onAuthStateChange={setIsSignedIn} />}
            {!isSignedIn && (
              <Link
                href="/simulate"
                className="btn-primary text-xs md:text-sm px-4 py-2 md:px-6 md:py-3"
              >
                今すぐ試す
              </Link>
            )}
          </nav>
        </div>
      </div>
    </header>
  )
}
