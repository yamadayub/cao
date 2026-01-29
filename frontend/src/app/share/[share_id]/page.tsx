import { Metadata } from 'next'
import { SnsShareViewClient } from './SnsShareViewClient'

interface SnsSharePageProps {
  params: Promise<{
    share_id: string
  }>
}

/**
 * OGPメタデータを動的に生成
 */
export async function generateMetadata({ params }: SnsSharePageProps): Promise<Metadata> {
  const { share_id } = await params
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  try {
    // サーバーサイドでシェアデータを取得
    const response = await fetch(`${apiUrl}/api/v1/share/${share_id}`, {
      next: { revalidate: 60 }, // 1分間キャッシュ
    })

    if (!response.ok) {
      return {
        title: 'シェア画像が見つかりません - Cao',
        description: 'この画像は期限切れか、削除された可能性があります。',
      }
    }

    const data = await response.json()

    if (!data.success) {
      return {
        title: 'シェア画像が見つかりません - Cao',
        description: 'この画像は期限切れか、削除された可能性があります。',
      }
    }

    const shareData = data.data
    const title = shareData.caption || 'Caoで作成したシミュレーション画像'
    const description = 'AIを使った顔シミュレーション結果をご覧ください。'

    return {
      title: `${title} - Cao`,
      description,
      openGraph: {
        title,
        description,
        images: [
          {
            url: shareData.share_image_url,
            width: shareData.template === 'before_after' ? 1200 : 1080,
            height: shareData.template === 'before_after' ? 630 : 1080,
            alt: title,
          },
        ],
        type: 'website',
        siteName: 'Cao',
      },
      twitter: {
        card: 'summary_large_image',
        title,
        description,
        images: [shareData.share_image_url],
      },
    }
  } catch {
    return {
      title: 'シェア画像 - Cao',
      description: 'AIを使った顔シミュレーション結果',
    }
  }
}

/**
 * SNSシェアページ
 *
 * UC-016: シェアページ閲覧
 */
export default async function SnsSharePage({ params }: SnsSharePageProps) {
  const { share_id } = await params

  return <SnsShareViewClient shareId={share_id} testId="sns-share-view" />
}
