'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'

export function LandingPageClient() {
  const [mounted, setMounted] = useState(false)

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
    <div className="min-h-screen flex flex-col bg-white">
      <Header variant="transparent" />

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center pt-20">
        {/* Background - Elegant gradient */}
        <div className="absolute inset-0 z-0">
          <div className="relative w-full h-full bg-gradient-to-br from-primary-50 via-white to-neutral-50">
            {/* Decorative elements */}
            <div className="absolute top-20 right-10 w-72 h-72 bg-primary-100/30 rounded-full blur-3xl"></div>
            <div className="absolute bottom-40 left-10 w-96 h-96 bg-primary-50/40 rounded-full blur-3xl"></div>
            {/* Gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-b from-white/30 via-transparent to-white"></div>
          </div>
        </div>

        {/* Content */}
        <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
          <div className="opacity-0 animate-fade-in-up">
            <p className="text-sm tracking-[0.3em] text-primary-600 uppercase mb-6">
              AI Face Simulation
            </p>
          </div>

          <h1 className="opacity-0 animate-fade-in-up animation-delay-100">
            <span className="block font-serif text-display-1 md:text-display-1-lg text-neutral-900 mb-2">
              理想の自分を、
            </span>
            <span className="block font-serif text-display-2 md:text-display-2-lg text-primary-700">
              AIでシミュレーション
            </span>
          </h1>

          <p className="opacity-0 animate-fade-in-up animation-delay-200 mt-6 text-base md:text-lg text-neutral-600 max-w-xl mx-auto leading-relaxed">
            あなたの顔写真と理想の顔を組み合わせて、
            <br className="hidden md:block" />
            段階的な変化をシミュレーションできます。
          </p>

          <div className="opacity-0 animate-fade-in-up animation-delay-300 mt-12 flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/simulate" className="btn-primary">
              無料で試す
            </Link>
            <a href="#how-it-works" className="btn-secondary">
              詳しく見る
            </a>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-10 left-1/2 transform -translate-x-1/2 opacity-0 animate-fade-in animation-delay-600">
          <a href="#how-it-works" className="flex flex-col items-center text-neutral-400 hover:text-primary-600 transition-colors duration-300">
            <span className="text-xs tracking-widest uppercase mb-2">Scroll</span>
            <svg className="w-5 h-5 animate-bounce-slow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </a>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="section-padding bg-white">
        <div className="container-narrow text-center">
          <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">How it works</p>
          <h2 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900 mb-12">使い方</h2>

          <div className="grid grid-cols-1 gap-10 md:gap-12">
            {/* Step 1 */}
            <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6 text-center md:text-left">
              <div className="flex-shrink-0 w-16 h-16 flex items-center justify-center">
                <span className="font-serif text-4xl md:text-5xl text-primary-200">01</span>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-xl md:text-2xl text-neutral-900 mb-2">
                  現在の顔をアップロード
                </h3>
                <p className="text-neutral-500 text-sm leading-relaxed">
                  正面を向いた明るい写真をアップロードしてください。
                </p>
              </div>
            </div>

            {/* Divider */}
            <div className="w-px h-8 bg-neutral-200 mx-auto"></div>

            {/* Step 2 */}
            <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6 text-center md:text-left">
              <div className="flex-shrink-0 w-16 h-16 flex items-center justify-center">
                <span className="font-serif text-4xl md:text-5xl text-primary-200">02</span>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-xl md:text-2xl text-neutral-900 mb-2">
                  理想の顔をアップロード
                </h3>
                <p className="text-neutral-500 text-sm leading-relaxed">
                  なりたい顔の写真をアップロードしてください。
                </p>
              </div>
            </div>

            {/* Divider */}
            <div className="w-px h-8 bg-neutral-200 mx-auto"></div>

            {/* Step 3 */}
            <div className="flex flex-col md:flex-row items-center gap-4 md:gap-6 text-center md:text-left">
              <div className="flex-shrink-0 w-16 h-16 flex items-center justify-center">
                <span className="font-serif text-4xl md:text-5xl text-primary-200">03</span>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-xl md:text-2xl text-neutral-900 mb-2">
                  結果を確認
                </h3>
                <p className="text-neutral-500 text-sm leading-relaxed">
                  スライダーで変化度を調整して確認できます。
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="section-padding bg-neutral-50">
        <div className="container-wide">
          <div className="text-center mb-10 md:mb-16">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">Features</p>
            <h2 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900">特徴</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Feature 1 */}
            <div className="bg-white p-6 md:p-8 rounded-2xl shadow-elegant">
              <div className="w-12 h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="font-serif text-lg md:text-xl text-neutral-900 mb-2">
                段階的な変化を確認
              </h3>
              <p className="text-neutral-500 text-sm leading-relaxed">
                0%から100%まで、5段階で徐々に変化する様子を確認できます。
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-white p-6 md:p-8 rounded-2xl shadow-elegant">
              <div className="w-12 h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </div>
              <h3 className="font-serif text-lg md:text-xl text-neutral-900 mb-2">
                スライダーで自由に調整
              </h3>
              <p className="text-neutral-500 text-sm leading-relaxed">
                好みの変化度をスライダーで簡単に選べます。
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-white p-6 md:p-8 rounded-2xl shadow-elegant">
              <div className="w-12 h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
              </div>
              <h3 className="font-serif text-lg md:text-xl text-neutral-900 mb-2">
                施術者と画像で共有
              </h3>
              <p className="text-neutral-500 text-sm leading-relaxed">
                結果をURLで共有して、施術者との相談に活用できます。
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Before/After Section */}
      <section className="section-padding bg-white">
        <div className="container-wide">
          <div className="text-center mb-10 md:mb-16">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">Result</p>
            <h2 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900">シミュレーション例</h2>
          </div>

          <div className="max-w-4xl mx-auto">
            <div className="grid grid-cols-2 gap-4 md:gap-8">
              {/* Before */}
              <div className="relative group">
                <div className="aspect-[3/4] bg-gradient-to-br from-neutral-100 to-neutral-200 shadow-elegant overflow-hidden relative">
                  <Image
                    src="/images/hero/before.png"
                    alt="Before"
                    fill
                    className="object-cover transition-transform duration-500 group-hover:scale-105 z-10"
                  />
                </div>
                <p className="absolute bottom-4 left-4 z-20 text-sm font-medium text-neutral-700 bg-white/90 px-4 py-2 backdrop-blur-sm shadow-sm">
                  Before
                </p>
              </div>

              {/* After */}
              <div className="relative group">
                <div className="aspect-[3/4] bg-gradient-to-br from-primary-50 to-primary-100 shadow-elegant overflow-hidden relative">
                  <Image
                    src="/images/hero/after.png"
                    alt="After"
                    fill
                    className="object-cover transition-transform duration-500 group-hover:scale-105 z-10"
                  />
                </div>
                <p className="absolute bottom-4 left-4 z-20 text-sm font-medium text-primary-700 bg-white/90 px-4 py-2 backdrop-blur-sm shadow-sm">
                  After
                </p>
              </div>
            </div>
            <p className="text-center text-sm text-neutral-400 mt-8">
              ※ シミュレーション結果はイメージです
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 md:py-24 bg-primary-700">
        <div className="container-narrow text-center">
          <h2 className="font-serif text-display-3 md:text-display-3-lg text-white mb-4">
            さあ、始めましょう
          </h2>
          <p className="text-primary-200 text-sm md:text-base mb-8 leading-relaxed">
            無料でシミュレーションを体験できます
            <br />
            理想の自分を見つけてみませんか？
          </p>
          <Link
            href="/simulate"
            className="inline-flex items-center justify-center px-8 py-3 text-sm font-medium text-primary-700 bg-white hover:bg-neutral-50 rounded-full transition-all duration-300 shadow-elegant"
          >
            今すぐ無料で試す
          </Link>
        </div>
      </section>

      <Footer />
    </div>
  )
}
