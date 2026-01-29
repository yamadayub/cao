/**
 * 保留アクション管理ユーティリティ
 *
 * UC-012: ログインして直前のアクションを継続
 *
 * ログインが必要なアクションを実行しようとした際に、
 * そのアクション情報を保存し、ログイン完了後に自動実行する。
 */

import type { PartsSelection } from '@/lib/api/types'

/**
 * 保留アクションの種類
 */
export type PendingActionType = 'parts-blur' | 'download' | 'save' | 'share' | 'sns-share'

/**
 * 保留アクションの情報
 */
export interface PendingAction {
  /** アクションの種類 */
  type: PendingActionType
  /** 現在の表示モード */
  viewMode: 'morph' | 'parts'
  /** パーツ選択状態（パーツモードの場合） */
  partsSelection?: PartsSelection
  /** タイムスタンプ */
  timestamp?: number
}

const STORAGE_KEY = 'cao_pending_action'
const MAX_AGE_MS = 5 * 60 * 1000 // 5分

/**
 * 保留アクションをsessionStorageに保存
 */
export function savePendingAction(action: PendingAction): void {
  if (typeof window === 'undefined') return

  const actionWithTimestamp: PendingAction = {
    ...action,
    timestamp: Date.now(),
  }

  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(actionWithTimestamp))
  } catch (error) {
    console.warn('Failed to save pending action:', error)
  }
}

/**
 * 保留アクションをsessionStorageから取得
 *
 * @returns 保留アクション、または存在しない/期限切れの場合はnull
 */
export function getPendingAction(): PendingAction | null {
  if (typeof window === 'undefined') return null

  try {
    const stored = sessionStorage.getItem(STORAGE_KEY)
    if (!stored) return null

    const action: PendingAction = JSON.parse(stored)

    // タイムスタンプがあり、期限切れの場合はクリア
    if (action.timestamp && Date.now() - action.timestamp > MAX_AGE_MS) {
      clearPendingAction()
      return null
    }

    return action
  } catch (error) {
    console.warn('Failed to get pending action:', error)
    return null
  }
}

/**
 * 保留アクションをsessionStorageから削除
 */
export function clearPendingAction(): void {
  if (typeof window === 'undefined') return

  try {
    sessionStorage.removeItem(STORAGE_KEY)
  } catch (error) {
    console.warn('Failed to clear pending action:', error)
  }
}

/**
 * アクションタイプに応じたモーダルのメッセージを取得
 */
export function getLoginPromptMessage(actionType: PendingActionType): {
  title: string
  description: string
} {
  switch (actionType) {
    case 'parts-blur':
      return {
        title: 'パーツ別結果を見るにはログインが必要です',
        description: 'ログインすると、パーツ別シミュレーションの結果をブラーなしで確認できます。',
      }
    case 'download':
      return {
        title: 'ダウンロードするにはログインが必要です',
        description: 'ログインすると、シミュレーション結果をダウンロードできます。',
      }
    case 'save':
      return {
        title: '保存するにはログインが必要です',
        description: 'ログインすると、シミュレーション結果を保存できます。',
      }
    case 'share':
      return {
        title: '共有するにはログインが必要です',
        description: 'ログインすると、シミュレーション結果を共有できます。',
      }
    case 'sns-share':
      return {
        title: 'SNSにシェアするにはログインが必要です',
        description: 'ログインすると、シミュレーション結果をSNSでシェアできます。',
      }
    default:
      return {
        title: 'ログインが必要です',
        description: 'この機能を使用するにはログインが必要です。',
      }
  }
}
