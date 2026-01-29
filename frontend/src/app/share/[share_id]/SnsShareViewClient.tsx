'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { getSnsShare, GetSnsShareData, ApiError } from '@/lib/api'

interface SnsShareViewClientProps {
  shareId: string
  testId?: string
}

type ViewState = 'loading' | 'success' | 'expired' | 'not_found' | 'error'

/**
 * SNSシェア閲覧クライアントコンポーネント
 */
export function SnsShareViewClient({ shareId, testId }: SnsShareViewClientProps) {
  const [viewState, setViewState] = useState<ViewState>('loading')
  const [shareData, setShareData] = useState<GetSnsShareData | null>(null)
  const [errorMessage, setErrorMessage] = useState('')

  useEffect(() => {
    async function fetchShareData() {
      try {
        const data = await getSnsShare(shareId)
        setShareData(data)

        if (data.is_expired) {
          setViewState('expired')
        } else {
          setViewState('success')
        }
      } catch (error) {
        if (error instanceof ApiError) {
          if (error.code === 'NOT_FOUND') {
            setViewState('not_found')
          } else {
            setViewState('error')
            setErrorMessage(error.localizedMessage)
          }
        } else {
          setViewState('error')
          setErrorMessage('データの取得に失敗しました')
        }
      }
    }

    fetchShareData()
  }, [shareId])

  if (viewState === 'loading') {
    return (
      <div
        data-testid={testId}
        className="min-h-screen flex items-center justify-center bg-gray-50"
      >
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">読み込み中...</p>
        </div>
      </div>
    )
  }

  if (viewState === 'not_found') {
    return (
      <div
        data-testid={testId}
        className="min-h-screen flex items-center justify-center bg-gray-50 p-4"
      >
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M12 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h1 className="text-xl font-bold text-gray-900 mb-2">
            ページが見つかりません
          </h1>
          <p className="text-gray-600 mb-6">
            お探しのシェア画像は存在しないか、削除された可能性があります。
          </p>
          <Link
            href="/"
            className="inline-block px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
          >
            トップページへ
          </Link>
        </div>
      </div>
    )
  }

  if (viewState === 'expired') {
    return (
      <div
        data-testid={testId}
        className="min-h-screen flex items-center justify-center bg-gray-50 p-4"
      >
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-amber-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h1 className="text-xl font-bold text-gray-900 mb-2">
            シェア期限が切れています
          </h1>
          <p className="text-gray-600 mb-6">
            このシェア画像の公開期限が終了しました。
          </p>
          <Link
            href="/"
            className="inline-block px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
          >
            Caoを試してみる
          </Link>
        </div>
      </div>
    )
  }

  if (viewState === 'error') {
    return (
      <div
        data-testid={testId}
        className="min-h-screen flex items-center justify-center bg-gray-50 p-4"
      >
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-red-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <h1 className="text-xl font-bold text-gray-900 mb-2">
            エラーが発生しました
          </h1>
          <p className="text-gray-600 mb-6">{errorMessage}</p>
          <Link
            href="/"
            className="inline-block px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
          >
            トップページへ
          </Link>
        </div>
      </div>
    )
  }

  // 成功時
  return (
    <div
      data-testid={testId}
      className="min-h-screen bg-gray-50"
    >
      {/* ヘッダー */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-2xl mx-auto px-4 py-4">
          <Link href="/" className="text-xl font-bold text-gray-900">
            Cao
          </Link>
        </div>
      </header>

      {/* コンテンツ */}
      <main className="max-w-2xl mx-auto px-4 py-8">
        {/* シェア画像 */}
        <div className="bg-white rounded-xl shadow-sm overflow-hidden mb-6">
          <img
            src={shareData?.share_image_url}
            alt="シェア画像"
            className="w-full"
            data-testid={testId ? `${testId}-image` : undefined}
          />
        </div>

        {/* キャプション */}
        {shareData?.caption && (
          <div
            className="bg-white rounded-xl shadow-sm p-4 mb-6"
            data-testid={testId ? `${testId}-caption` : undefined}
          >
            <p className="text-gray-700">{shareData.caption}</p>
          </div>
        )}

        {/* CTA */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl p-6 text-center text-white">
          <h2 className="text-xl font-bold mb-2">
            あなたも試してみませんか？
          </h2>
          <p className="text-blue-100 mb-4">
            AIが理想の顔をシミュレーションします
          </p>
          <Link
            href="/simulate"
            className="inline-block px-6 py-3 bg-white text-blue-600 font-semibold rounded-lg hover:bg-blue-50 transition-colors"
            data-testid={testId ? `${testId}-cta` : undefined}
          >
            無料で試す
          </Link>
        </div>

        {/* フッター情報 */}
        <div className="text-center mt-8 text-sm text-gray-500">
          <p>
            作成日: {shareData && new Date(shareData.created_at).toLocaleDateString('ja-JP')}
          </p>
        </div>
      </main>
    </div>
  )
}
