'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { useAuth, useClerk, SignInButton } from '@clerk/nextjs'

interface HeaderProps {
  variant?: 'default' | 'transparent'
}

function HeaderAuthSection() {
  const { isLoaded, isSignedIn } = useAuth()
  const { loaded: clerkLoaded } = useClerk()

  // Clerkがまだロードされていない場合
  if (!isLoaded || !clerkLoaded) {
    return (
      <div className="w-8 h-8 rounded-full bg-neutral-100 animate-pulse" />
    )
  }

  // 常に「ログイン」ボタンを表示
  // - 未認証: クリックでログインモーダル表示
  // - 認証済み: クリックでマイページへ遷移
  return (
    <>
      {isSignedIn ? (
        <Link
          href="/mypage"
          className="btn-primary text-xs md:text-sm px-4 py-2 md:px-6 md:py-3"
        >
          ログイン
        </Link>
      ) : (
        <SignInButton mode="modal">
          <button className="btn-primary text-xs md:text-sm px-4 py-2 md:px-6 md:py-3">
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
            {mounted && <HeaderAuthSection />}
          </nav>
        </div>
      </div>
    </header>
  )
}
