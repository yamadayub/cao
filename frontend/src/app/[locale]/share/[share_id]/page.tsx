import { Metadata } from 'next'
import { setRequestLocale, getTranslations } from 'next-intl/server'
import { SnsShareViewClient } from './SnsShareViewClient'

interface SnsSharePageProps {
  params: Promise<{
    locale: string
    share_id: string
  }>
}

/**
 * OGPメタデータを動的に生成
 */
export async function generateMetadata({ params }: SnsSharePageProps): Promise<Metadata> {
  const { locale, share_id } = await params;
  const t = await getTranslations({ locale, namespace: 'share' })
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  try {
    // サーバーサイドでシェアデータを取得
    const response = await fetch(`${apiUrl}/api/v1/share/${share_id}`, {
      next: { revalidate: 60 }, // 1分間キャッシュ
    })

    if (!response.ok) {
      return {
        title: t('metadata.snsNotFoundTitle'),
        description: t('metadata.snsNotFoundDescription'),
      }
    }

    const data = await response.json()

    if (!data.success) {
      return {
        title: t('metadata.snsNotFoundTitle'),
        description: t('metadata.snsNotFoundDescription'),
      }
    }

    const shareData = data.data
    const title = shareData.caption || t('metadata.snsDefaultTitle')
    const description = t('metadata.snsDefaultDescription')

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
      title: t('metadata.snsFallbackTitle'),
      description: t('metadata.snsFallbackDescription'),
    }
  }
}

/**
 * SNSシェアページ
 *
 * UC-016: シェアページ閲覧
 */
export default async function SnsSharePage({ params }: SnsSharePageProps) {
  const { locale, share_id } = await params
  setRequestLocale(locale)

  return <SnsShareViewClient shareId={share_id} testId="sns-share-view" />
}
