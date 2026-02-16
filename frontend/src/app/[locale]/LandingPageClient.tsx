'use client'

import { useState, useEffect } from 'react'
import { useTranslations } from 'next-intl'
import { Link } from '@/i18n/navigation'
import Image from 'next/image'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'

export function LandingPageClient() {
  const [mounted, setMounted] = useState(false)
  const t = useTranslations('landing')
  const tc = useTranslations('common')

  useEffect(() => {
    setMounted(true)
  }, [])

  // ローディング表示
  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="w-8 h-8 border-2 border-primary-200 border-t-primary-700 rounded-full animate-spin"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-white overflow-x-hidden">
      <Header variant="transparent" />

      {/* Hero Section */}
      <section className="relative min-h-[100svh] flex items-center justify-center pt-16 pb-8 overflow-hidden">
        {/* Background - Elegant gradient */}
        <div className="absolute inset-0 z-0 overflow-hidden">
          <div className="relative w-full h-full bg-gradient-to-br from-primary-50 via-white to-neutral-50">
            <div className="hidden md:block absolute top-20 right-0 w-72 h-72 bg-primary-100/30 rounded-full blur-3xl"></div>
            <div className="hidden md:block absolute bottom-40 left-0 w-72 h-72 bg-primary-50/40 rounded-full blur-3xl"></div>
            <div className="absolute inset-0 bg-gradient-to-b from-white/30 via-transparent to-white"></div>
          </div>
        </div>

        {/* Content */}
        <div className="relative z-10 max-w-4xl mx-auto px-4 md:px-6 text-center">
          <div className="opacity-0 animate-fade-in-up">
            <p className="text-xs tracking-[0.3em] text-primary-600 uppercase mb-4 md:mb-6">
              {t('hero.subtitle')}
            </p>
          </div>

          <h1 className="opacity-0 animate-fade-in-up animation-delay-100">
            <span className="block font-serif text-2xl md:text-display-1-lg text-neutral-900 mb-1 md:mb-2">
              {t('hero.title1')}
            </span>
            <span className="block font-serif text-2xl md:text-display-1-lg text-neutral-900 mb-1 md:mb-2">
              {t('hero.title2')}
            </span>
            <span className="block font-serif text-xl md:text-display-2-lg text-primary-700 mt-2 md:mt-4">
              {t('hero.title3')}
            </span>
          </h1>

          <div className="opacity-0 animate-fade-in-up animation-delay-300 mt-8 md:mt-12 flex flex-col sm:flex-row gap-3 md:gap-4 justify-center">
            <Link href="/simulate" className="btn-primary">
              {tc('actions.tryFree')}
            </Link>
            <a href="#how-it-works" className="btn-secondary">
              {tc('actions.learnMore')}
            </a>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-6 md:bottom-10 left-1/2 transform -translate-x-1/2 opacity-0 animate-fade-in animation-delay-600">
          <a href="#how-it-works" className="flex flex-col items-center text-neutral-400 hover:text-primary-600 transition-colors duration-300">
            <span className="text-xs tracking-widest uppercase mb-1">{t('hero.scroll')}</span>
            <svg className="w-4 h-4 md:w-5 md:h-5 animate-bounce-slow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </a>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="min-h-[100svh] flex flex-col justify-center py-8 md:py-16 bg-white">
        <div className="container-narrow text-center px-4">
          <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-2">{t('howItWorks.subtitle')}</p>
          <h2 className="font-serif text-2xl md:text-display-3-lg text-neutral-900 mb-6 md:mb-10">{t('howItWorks.title')}</h2>

          <div className="grid grid-cols-1 gap-4 md:gap-8">
            {/* Step 1 */}
            <div className="flex flex-col md:flex-row items-center gap-3 md:gap-6 text-center md:text-left bg-neutral-50 rounded-xl p-4 md:p-6">
              <div className="flex-shrink-0 w-12 h-12 md:w-16 md:h-16 flex items-center justify-center">
                <span className="font-serif text-3xl md:text-5xl text-primary-200">01</span>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-lg md:text-2xl text-neutral-900 mb-1">
                  {t('howItWorks.step1Title')}
                </h3>
                <p className="text-neutral-500 text-xs md:text-sm leading-relaxed">
                  {t('howItWorks.step1Desc')}
                </p>
              </div>
            </div>

            <div className="flex justify-center">
              <svg className="w-5 h-5 text-primary-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>

            {/* Step 2 */}
            <div className="flex flex-col md:flex-row items-center gap-3 md:gap-6 text-center md:text-left bg-neutral-50 rounded-xl p-4 md:p-6">
              <div className="flex-shrink-0 w-12 h-12 md:w-16 md:h-16 flex items-center justify-center">
                <span className="font-serif text-3xl md:text-5xl text-primary-200">02</span>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-lg md:text-2xl text-neutral-900 mb-1">
                  {t('howItWorks.step2Title')}
                </h3>
                <p className="text-neutral-500 text-xs md:text-sm leading-relaxed">
                  {t('howItWorks.step2Desc')}
                </p>
              </div>
            </div>

            <div className="flex justify-center">
              <svg className="w-5 h-5 text-primary-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>

            {/* Step 3 */}
            <div className="flex flex-col md:flex-row items-center gap-3 md:gap-6 text-center md:text-left bg-neutral-50 rounded-xl p-4 md:p-6">
              <div className="flex-shrink-0 w-12 h-12 md:w-16 md:h-16 flex items-center justify-center">
                <span className="font-serif text-3xl md:text-5xl text-primary-200">03</span>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-lg md:text-2xl text-neutral-900 mb-1">
                  {t('howItWorks.step3Title')}
                </h3>
                <p className="text-neutral-500 text-xs md:text-sm leading-relaxed">
                  {t('howItWorks.step3Desc')}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="min-h-[100svh] flex flex-col justify-center py-8 md:py-16 bg-neutral-50">
        <div className="container-wide px-4">
          <div className="text-center mb-6 md:mb-12">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-2">{t('features.subtitle')}</p>
            <h2 className="font-serif text-2xl md:text-display-3-lg text-neutral-900">{t('features.title')}</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 max-w-4xl mx-auto">
            {/* Feature 1 */}
            <div className="bg-white p-5 md:p-8 rounded-2xl shadow-elegant text-center">
              <div className="w-10 h-10 md:w-12 md:h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-3 md:mb-4 mx-auto">
                <svg className="w-5 h-5 md:w-6 md:h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
                </svg>
              </div>
              <h3 className="font-serif text-base md:text-xl text-neutral-900 mb-1 md:mb-2">
                {t('features.aiTitle')}
              </h3>
              <p className="text-neutral-500 text-xs md:text-sm leading-relaxed">
                {t('features.aiDesc')}
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-white p-5 md:p-8 rounded-2xl shadow-elegant text-center">
              <div className="w-10 h-10 md:w-12 md:h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-3 md:mb-4 mx-auto">
                <svg className="w-5 h-5 md:w-6 md:h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
                </svg>
              </div>
              <h3 className="font-serif text-base md:text-xl text-neutral-900 mb-1 md:mb-2">
                {t('features.partsTitle')}
              </h3>
              <p className="text-neutral-500 text-xs md:text-sm leading-relaxed">
                {t('features.partsDesc')}
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-white p-5 md:p-8 rounded-2xl shadow-elegant text-center">
              <div className="w-10 h-10 md:w-12 md:h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-3 md:mb-4 mx-auto">
                <svg className="w-5 h-5 md:w-6 md:h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                </svg>
              </div>
              <h3 className="font-serif text-base md:text-xl text-neutral-900 mb-1 md:mb-2">
                {t('features.compareTitle')}
              </h3>
              <p className="text-neutral-500 text-xs md:text-sm leading-relaxed">
                {t('features.compareDesc')}
              </p>
            </div>

            {/* Feature 4 */}
            <div className="bg-white p-5 md:p-8 rounded-2xl shadow-elegant text-center">
              <div className="w-10 h-10 md:w-12 md:h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-3 md:mb-4 mx-auto">
                <svg className="w-5 h-5 md:w-6 md:h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                </svg>
              </div>
              <h3 className="font-serif text-base md:text-xl text-neutral-900 mb-1 md:mb-2">
                {t('features.privacyTitle')}
              </h3>
              <p className="text-neutral-500 text-xs md:text-sm leading-relaxed">
                {t('features.privacyDesc')}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Before/After Section */}
      <section className="py-12 md:py-20 bg-white">
        <div className="container-wide px-4">
          <div className="text-center mb-6 md:mb-12">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-2">{t('examples.subtitle')}</p>
            <h2 className="font-serif text-2xl md:text-display-3-lg text-neutral-900">{t('examples.title')}</h2>
          </div>

          <div className="max-w-3xl mx-auto">
            <div className="grid grid-cols-2 gap-3 md:gap-8">
              {/* Before */}
              <div className="relative group">
                <div className="aspect-[3/4] bg-gradient-to-br from-neutral-100 to-neutral-200 shadow-elegant overflow-hidden relative rounded-lg md:rounded-xl">
                  <Image
                    src="/images/hero/before.png"
                    alt="Before"
                    fill
                    className="object-cover transition-transform duration-500 group-hover:scale-105 z-10"
                  />
                </div>
                <p className="absolute bottom-2 left-2 md:bottom-4 md:left-4 z-20 text-xs md:text-sm font-medium text-neutral-700 bg-white/90 px-2 py-1 md:px-4 md:py-2 rounded backdrop-blur-sm shadow-sm">
                  Before
                </p>
              </div>

              {/* After */}
              <div className="relative group">
                <div className="aspect-[3/4] bg-gradient-to-br from-primary-50 to-primary-100 shadow-elegant overflow-hidden relative rounded-lg md:rounded-xl">
                  <Image
                    src="/images/hero/after.png"
                    alt="After"
                    fill
                    className="object-cover transition-transform duration-500 group-hover:scale-105 z-10"
                  />
                </div>
                <p className="absolute bottom-2 left-2 md:bottom-4 md:left-4 z-20 text-xs md:text-sm font-medium text-primary-700 bg-white/90 px-2 py-1 md:px-4 md:py-2 rounded backdrop-blur-sm shadow-sm">
                  After
                </p>
              </div>
            </div>
            <p className="text-center text-xs md:text-sm text-neutral-400 mt-4 md:mt-8">
              {t('examples.disclaimer')}
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-12 md:py-20 bg-primary-700">
        <div className="container-narrow text-center px-4">
          <h2 className="font-serif text-2xl md:text-display-3-lg text-white mb-3 md:mb-4">
            {t('cta.title')}
          </h2>
          <p className="text-primary-200 text-xs md:text-base mb-6 md:mb-8 leading-relaxed">
            {t('cta.desc1')}
            <br />
            {t('cta.desc2')}
          </p>
          <Link
            href="/simulate"
            className="inline-flex items-center justify-center px-6 md:px-8 py-2.5 md:py-3 text-sm font-medium text-primary-700 bg-white hover:bg-neutral-50 rounded-full transition-all duration-300 shadow-elegant"
          >
            {tc('actions.tryNow')}
          </Link>
        </div>
      </section>

      <Footer />
    </div>
  )
}
