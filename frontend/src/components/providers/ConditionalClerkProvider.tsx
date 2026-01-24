'use client'

import { ClerkProvider } from '@clerk/nextjs'
import { ReactNode } from 'react'

interface ConditionalClerkProviderProps {
  children: ReactNode
}

/**
 * Clerk認証プロバイダーのラッパー
 *
 * Clerk APIキーが設定されていない場合（テスト環境など）は
 * ClerkProviderをスキップしてそのままchildrenをレンダリングする
 */
export function ConditionalClerkProvider({ children }: ConditionalClerkProviderProps) {
  const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY

  // Clerk キーが設定されていない場合はClerkProviderをスキップ
  if (!publishableKey) {
    return <>{children}</>
  }

  return <ClerkProvider publishableKey={publishableKey}>{children}</ClerkProvider>
}
