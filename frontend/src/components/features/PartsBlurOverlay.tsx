'use client'

import { useCallback } from 'react'

/**
 * パーツ名のマッピング
 */
const PARTS_NAMES: Record<string, string> = {
  // シンプル3択
  eyes: '目',
  nose: '鼻',
  lips: '唇',
  // 個別パーツ
  left_eye: '左目',
  right_eye: '右目',
  left_eyebrow: '左眉',
  right_eyebrow: '右眉',
}

/**
 * ログイン誘導コールバックに渡される情報
 */
export interface LoginPromptInfo {
  title: string
  description: string
}

/**
 * PartsBlurOverlayコンポーネントのProps
 */
export interface PartsBlurOverlayProps {
  /** 表示する画像のURL (Base64 Data URLまたはURL) */
  imageUrl: string
  /** 画像のalt属性 */
  altText: string
  /** 認証状態 */
  isAuthenticated?: boolean
  /** ログインクリック時のコールバック */
  onLoginClick?: (info?: LoginPromptInfo) => void
  /** ローディング状態 */
  isLoading?: boolean
  /** カスタム幅 */
  width?: number
  /** カスタム高さ */
  height?: number
  /** パーツラベルを表示するか */
  showPartsLabel?: boolean
  /** 適用されたパーツのリスト */
  appliedParts?: string[]
  /** ログイン誘導モーダルのタイトル */
  loginPromptTitle?: string
  /** ログイン誘導モーダルの説明文 */
  loginPromptDescription?: string
  /** テスト用のdata-testid */
  testId?: string
}

/**
 * パーツ別シミュレーション結果のブラー表示コンポーネント
 *
 * - 未認証ユーザーには画像にブラーを適用
 * - 認証済みユーザーにはブラーなしで表示
 * - 未認証時にタップでログイン誘導コールバックを実行
 *
 * @see functional-spec.md セクション 3.4 (SCR-003) パーツ別適用モード
 */
export function PartsBlurOverlay({
  imageUrl,
  altText,
  isAuthenticated = false,
  onLoginClick,
  isLoading = false,
  width,
  height,
  showPartsLabel = false,
  appliedParts = [],
  loginPromptTitle = 'パーツ別の結果を見るにはログインが必要です',
  loginPromptDescription = 'パーツ別シミュレーションの詳細な結果を確認するにはログインしてください。',
  testId = 'parts-blur-overlay',
}: PartsBlurOverlayProps) {
  /**
   * クリックハンドラ
   */
  const handleClick = useCallback(() => {
    if (!isAuthenticated && onLoginClick) {
      onLoginClick({
        title: loginPromptTitle,
        description: loginPromptDescription,
      })
    }
  }, [isAuthenticated, onLoginClick, loginPromptTitle, loginPromptDescription])

  /**
   * キーボードイベントハンドラ (アクセシビリティ対応)
   */
  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (!isAuthenticated && onLoginClick && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault()
        onLoginClick({
          title: loginPromptTitle,
          description: loginPromptDescription,
        })
      }
    },
    [isAuthenticated, onLoginClick, loginPromptTitle, loginPromptDescription]
  )

  /**
   * パーツ名のリストを日本語表示名に変換
   */
  const getPartsDisplayNames = () => {
    return appliedParts
      .map((part) => PARTS_NAMES[part] || part)
      .join(', ')
  }

  // スタイルオブジェクト
  const containerStyle: React.CSSProperties = {}
  if (width) containerStyle.width = `${width}px`
  if (height) containerStyle.height = `${height}px`

  // ローディング状態
  if (isLoading) {
    return (
      <div
        className="relative w-full h-full bg-neutral-200 animate-pulse rounded-xl"
        style={containerStyle}
        data-testid={`${testId}-skeleton`}
      >
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-10 h-10 border-2 border-neutral-300 border-t-neutral-600 rounded-full animate-spin" />
        </div>
      </div>
    )
  }

  // 画像URLがない場合のプレースホルダー
  if (!imageUrl) {
    return (
      <div
        className="relative w-full h-full bg-neutral-100 rounded-xl flex items-center justify-center"
        style={containerStyle}
        data-testid={`${testId}-placeholder`}
      >
        <div className="text-neutral-400 text-center p-4">
          <svg
            className="mx-auto h-12 w-12 mb-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p className="text-sm">画像がありません</p>
        </div>
      </div>
    )
  }

  // 認証済みユーザー: ブラーなしで表示
  if (isAuthenticated) {
    return (
      <div
        className="relative w-full h-full"
        style={containerStyle}
        data-testid={testId}
      >
        <div
          className="w-full h-full overflow-hidden rounded-xl"
          data-testid={`${testId}-image-container`}
        >
          <img
            src={imageUrl}
            alt={altText}
            className="w-full h-full object-cover"
          />
        </div>
        {/* パーツラベル表示 */}
        {showPartsLabel && appliedParts.length > 0 && (
          <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent">
            <p className="text-white text-sm text-center">
              適用パーツ: {getPartsDisplayNames()}
            </p>
          </div>
        )}
      </div>
    )
  }

  // 未認証ユーザー: ブラー適用
  return (
    <div
      className="relative w-full h-full cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 rounded-xl overflow-hidden"
      style={containerStyle}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-label="ログインしてパーツ別シミュレーション結果を表示"
      data-testid={testId}
    >
      {/* ブラー適用された画像 */}
      <div
        className="w-full h-full overflow-hidden blur-lg"
        data-testid={`${testId}-image-container`}
      >
        <img
          src={imageUrl}
          alt={altText}
          className="w-full h-full object-cover"
        />
      </div>

      {/* ブラーオーバーレイ */}
      <div
        className="absolute inset-0 bg-black/30 flex flex-col items-center justify-center blur"
        data-testid={`${testId}-blur`}
      >
        {/* このdivはブラー検出用のマーカー */}
      </div>

      {/* タップしてログインのテキスト (ブラーなし) */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <div className="bg-white/90 backdrop-blur-sm rounded-full px-6 py-3 shadow-lg">
          <p className="text-neutral-800 font-medium text-sm">
            タップしてログイン
          </p>
        </div>
      </div>

      {/* パーツラベル表示 */}
      {showPartsLabel && appliedParts.length > 0 && (
        <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/60 to-transparent pointer-events-none">
          <p className="text-white text-sm text-center">
            適用パーツ: {getPartsDisplayNames()}
          </p>
        </div>
      )}
    </div>
  )
}
