'use client'

import { SignUp } from '@clerk/nextjs'
import { useTranslations } from 'next-intl'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'

export default function SignUpPage() {
  const t = useTranslations('auth')

  return (
    <div className="min-h-screen flex flex-col bg-neutral-50">
      <Header />

      <main className="flex-1 flex flex-col items-center justify-center pt-20 pb-12 px-4">
        <div className="text-center mb-8">
          <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">Sign Up</p>
          <h1 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900">
            {t('signUp.title')}
          </h1>
        </div>
        <SignUp />
      </main>

      <Footer />
    </div>
  )
}
