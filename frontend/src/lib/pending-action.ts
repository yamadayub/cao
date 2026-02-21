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
  /** パーツ表示モード（パーツモードの場合） */
  partsViewMode?: 'current' | 'applied' | 'slider' | 'morphing'
  /** タイムスタンプ */
  timestamp?: number
}

const STORAGE_KEY = 'cao_pending_action'
const SWAPPED_IMAGE_KEY = 'cao_swapped_image'
const PARTS_BLEND_IMAGE_KEY = 'cao_parts_blend_image'
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
 * シミュレーション結果の画像データを保存
 * （ログイン後に復元するため）
 */
export function saveSimulationImages(data: {
  swappedImage?: string | null
  partsBlendImage?: string | null
}): void {
  if (typeof window === 'undefined') return

  try {
    if (data.swappedImage) {
      sessionStorage.setItem(SWAPPED_IMAGE_KEY, data.swappedImage)
    }
    if (data.partsBlendImage) {
      sessionStorage.setItem(PARTS_BLEND_IMAGE_KEY, data.partsBlendImage)
    }
  } catch (error) {
    console.warn('Failed to save simulation images:', error)
  }
}

/**
 * 保存されたシミュレーション結果の画像データを取得
 */
export function getSimulationImages(): {
  swappedImage: string | null
  partsBlendImage: string | null
} {
  if (typeof window === 'undefined') {
    return { swappedImage: null, partsBlendImage: null }
  }

  try {
    return {
      swappedImage: sessionStorage.getItem(SWAPPED_IMAGE_KEY),
      partsBlendImage: sessionStorage.getItem(PARTS_BLEND_IMAGE_KEY),
    }
  } catch (error) {
    console.warn('Failed to get simulation images:', error)
    return { swappedImage: null, partsBlendImage: null }
  }
}

/**
 * 保存されたシミュレーション結果の画像データを削除
 */
export function clearSimulationImages(): void {
  if (typeof window === 'undefined') return

  try {
    sessionStorage.removeItem(SWAPPED_IMAGE_KEY)
    sessionStorage.removeItem(PARTS_BLEND_IMAGE_KEY)
  } catch (error) {
    console.warn('Failed to clear simulation images:', error)
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
