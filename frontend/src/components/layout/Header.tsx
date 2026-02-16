'use client'

import { useState, useEffect } from 'react'
import { useTranslations } from 'next-intl'
import { Link } from '@/i18n/navigation'
import { useAuth, useClerk, SignInButton, UserButton } from '@clerk/nextjs'
import { isClerkAvailable } from '@/hooks/useClerkSafe'
import { LanguageSwitcher } from './LanguageSwitcher'

interface HeaderProps {
  variant?: 'default' | 'transparent'
}

function ClerkAuthSection() {
  const t = useTranslations('common')
  const { isLoaded, isSignedIn } = useAuth()
  const { loaded: clerkLoaded } = useClerk()

  // Clerkがまだロードされていない場合
  if (!isLoaded || !clerkLoaded) {
    return (
      <div className="w-8 h-8 rounded-full bg-neutral-100 animate-pulse" />
    )
  }

  return (
    <>
      {isSignedIn ? (
        <UserButton
          afterSignOutUrl="/"
          appearance={{
            elements: {
              avatarBox: 'w-10 h-10',
            },
          }}
        >
          <UserButton.MenuItems>
            <UserButton.Link
              label={t('nav.mypage')}
              labelIcon={<span>📋</span>}
              href="/mypage"
            />
          </UserButton.MenuItems>
        </UserButton>
      ) : (
        <SignInButton mode="modal">
          <button className="btn-primary text-xs md:text-sm px-4 py-2 md:px-6 md:py-3">
            {t('nav.login')}
          </button>
        </SignInButton>
      )}
    </>
  )
}

function FallbackAuthSection() {
  const t = useTranslations('common')

  return (
    <Link href="/sign-in">
      <button className="btn-primary text-xs md:text-sm px-4 py-2 md:px-6 md:py-3">
        {t('nav.login')}
      </button>
    </Link>
  )
}

function HeaderAuthSection() {
  if (!isClerkAvailable()) {
    return <FallbackAuthSection />
  }

  return <ClerkAuthSection />
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
          <nav className="flex items-center gap-3 md:gap-4">
            <LanguageSwitcher />
            {mounted && <HeaderAuthSection />}
          </nav>
        </div>
      </div>
    </header>
  )
}
