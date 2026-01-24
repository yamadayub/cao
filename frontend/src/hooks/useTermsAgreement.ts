'use client'

import { useCallback, useEffect, useState } from 'react'

/**
 * 同意状態の型定義
 */
export interface TermsAgreementState {
  agreed: boolean
  version: string
  timestamp: string
}

/**
 * localStorage のキー
 */
const STORAGE_KEY = 'cao_terms_agreed'

/**
 * 現在の利用規約バージョン
 */
const CURRENT_VERSION = '1.0'

/**
 * 利用規約同意状態を管理するフック
 *
 * localStorage に同意状態を保存し、同意済みかどうかをチェック、同意する関数を提供
 *
 * @returns {
 *   hasAgreed: boolean - 同意済みかどうか
 *   isLoading: boolean - ローディング中かどうか
 *   agree: () => void - 同意する関数
 *   agreementState: TermsAgreementState | null - 同意状態の詳細
 * }
 */
export function useTermsAgreement() {
  const [agreementState, setAgreementState] = useState<TermsAgreementState | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  /**
   * localStorage から同意状態を読み込む
   */
  useEffect(() => {
    try {
      const storedValue = localStorage.getItem(STORAGE_KEY)
      if (storedValue) {
        const parsed: TermsAgreementState = JSON.parse(storedValue)
        // バージョンが一致する場合のみ有効
        if (parsed.version === CURRENT_VERSION && parsed.agreed) {
          setAgreementState(parsed)
        }
      }
    } catch (error) {
      console.error('Failed to load terms agreement state:', error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  /**
   * 利用規約に同意する
   */
  const agree = useCallback(() => {
    const newState: TermsAgreementState = {
      agreed: true,
      version: CURRENT_VERSION,
      timestamp: new Date().toISOString(),
    }

    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newState))
      setAgreementState(newState)
    } catch (error) {
      console.error('Failed to save terms agreement state:', error)
    }
  }, [])

  /**
   * 同意済みかどうか
   */
  const hasAgreed = agreementState?.agreed === true && agreementState?.version === CURRENT_VERSION

  return {
    hasAgreed,
    isLoading,
    agree,
    agreementState,
    currentVersion: CURRENT_VERSION,
  }
}
