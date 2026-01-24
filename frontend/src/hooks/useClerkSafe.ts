'use client'

/* eslint-disable react-hooks/rules-of-hooks */
// Note: This file intentionally breaks hooks rules for conditional Clerk usage.
// When NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is not set, Clerk hooks are not called.
// This allows the app to work without Clerk in development/testing environments.

import { useMemo } from 'react'

/**
 * Clerkが利用可能かどうかを判定
 */
export function isClerkAvailable(): boolean {
  return !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
}

/**
 * Clerkのユーザー情報（Clerkなしでも動作する）
 */
interface SafeUser {
  isSignedIn: boolean
  user: {
    id?: string
    primaryEmailAddress?: { emailAddress: string } | null
  } | null
}

/**
 * Clerkの認証情報（Clerkなしでも動作する）
 */
interface SafeAuth {
  getToken: () => Promise<string | null>
  isLoaded: boolean
  userId: string | null
}

/**
 * Clerkが利用可能な場合はuseUserを使用、そうでなければデフォルト値を返す
 */
export function useUserSafe(): SafeUser {
  const clerkAvailable = isClerkAvailable()

  // Clerkが利用可能な場合のみuseUserを呼び出す
  const clerkUser = useMemo(() => {
    if (clerkAvailable && typeof window !== 'undefined') {
      try {
        // 動的にインポートしてuseUserを使用することはできないため、
        // Clerkが利用できない場合はデフォルト値を返す
        const { useUser } = require('@clerk/nextjs')
        return useUser()
      } catch {
        return null
      }
    }
    return null
  }, [clerkAvailable])

  if (!clerkAvailable || !clerkUser) {
    return {
      isSignedIn: false,
      user: null,
    }
  }

  return {
    isSignedIn: clerkUser.isSignedIn,
    user: clerkUser.user,
  }
}

/**
 * Clerkが利用可能な場合はuseAuthを使用、そうでなければデフォルト値を返す
 */
export function useAuthSafe(): SafeAuth {
  const clerkAvailable = isClerkAvailable()

  const clerkAuth = useMemo(() => {
    if (clerkAvailable && typeof window !== 'undefined') {
      try {
        const { useAuth } = require('@clerk/nextjs')
        return useAuth()
      } catch {
        return null
      }
    }
    return null
  }, [clerkAvailable])

  if (!clerkAvailable || !clerkAuth) {
    return {
      getToken: async () => null,
      isLoaded: true,
      userId: null,
    }
  }

  return {
    getToken: clerkAuth.getToken,
    isLoaded: clerkAuth.isLoaded,
    userId: clerkAuth.userId,
  }
}
