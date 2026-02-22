'use client'

import { ClerkProvider } from '@clerk/nextjs'
import { jaJP, enUS, zhCN, zhTW, koKR } from '@clerk/localizations'
import { ReactNode } from 'react'
import type { Locale } from '@/i18n/config'

const clerkLocales: Record<Locale, typeof jaJP> = {
  ja: jaJP,
  en: enUS,
  'zh-CN': zhCN,
  'zh-TW': zhTW,
  ko: koKR,
}

const clerkAppearance = {
  variables: {
    colorPrimary: '#8b6f7a',
    fontFamily: "'Noto Sans JP', sans-serif",
    borderRadius: '0.75rem',
  },
  elements: {
    formButtonPrimary: {
      borderRadius: '9999px',
    },
    headerTitle: {
      fontFamily: "'Cormorant Garamond', serif",
    },
    card: {
      boxShadow: '0 4px 24px rgba(0, 0, 0, 0.08)',
    },
    footerActionLink: {
      color: '#8b6f7a',
    },
  },
} as const

interface ConditionalClerkProviderProps {
  children: ReactNode
  locale?: string
}

/**
 * Clerk認証プロバイダーのラッパー
 *
 * Clerk APIキーが設定されていない場合（テスト環境など）は
 * ClerkProviderをスキップしてそのままchildrenをレンダリングする
 */
export function ConditionalClerkProvider({ children, locale }: ConditionalClerkProviderProps) {
  const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY

  // Clerk キーが設定されていない場合はClerkProviderをスキップ
  if (!publishableKey) {
    return <>{children}</>
  }

  const localization = clerkLocales[(locale as Locale)] ?? clerkLocales.ja

  return (
    <ClerkProvider
      publishableKey={publishableKey}
      localization={localization as any}
      appearance={clerkAppearance}
    >
      {children}
    </ClerkProvider>
  )
}
