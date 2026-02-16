import type { Metadata } from 'next';
import { Link } from '@/i18n/navigation'
import { getTranslations, setRequestLocale } from 'next-intl/server'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'

type Props = {
  params: Promise<{ locale: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'metadata' });

  return {
    title: t('privacy.title'),
    description: t('privacy.description'),
  };
}

const sectionKeys = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11'] as const;

/**
 * プライバシーポリシーページ (SCR-008)
 *
 * 参照: functional-spec.md セクション 3.1
 */
export default async function PrivacyPage({ params }: Props) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations('privacy');

  return (
    <div className="min-h-screen flex flex-col bg-neutral-50">
      <Header />

      <main className="flex-1 pt-20">
        <div className="container-narrow py-8 md:py-12">
          {/* ページタイトル */}
          <div className="text-center mb-8 md:mb-12">
            <p className="text-xs tracking-[0.2em] text-primary-600 uppercase mb-3">{t('subtitle')}</p>
            <h1 className="font-serif text-display-3 md:text-display-3-lg text-neutral-900">
              {t('title')}
            </h1>
          </div>

          {/* コンテンツ */}
          <article className="bg-white rounded-2xl shadow-elegant p-6 md:p-8">
            <div className="prose prose-neutral max-w-none">
              <p className="text-sm text-neutral-500 mb-6">{t('lastUpdated', { date: '2025-01-24' })}</p>

              <p className="text-neutral-700 leading-relaxed mb-8">
                {t('intro')}
              </p>

              {sectionKeys.map((key) => (
                <section key={key} className="mb-8">
                  <h2 className="text-xl font-serif text-neutral-900 mb-4">{t(`sections.${key}.title`)}</h2>
                  <p className="text-neutral-700 leading-relaxed whitespace-pre-line">{t(`sections.${key}.content`)}</p>
                </section>
              ))}
            </div>

            {/* フッターリンク */}
            <div className="mt-8 pt-6 border-t border-neutral-200">
              <div className="flex flex-wrap gap-4 text-sm">
                <Link href="/" className="text-primary-600 hover:text-primary-800">
                  {t('footer.backToTop')}
                </Link>
                <Link href="/terms" className="text-primary-600 hover:text-primary-800">
                  {t('footer.terms')}
                </Link>
                <Link href="/simulate" className="text-primary-600 hover:text-primary-800">
                  {t('footer.trySimulation')}
                </Link>
              </div>
            </div>
          </article>
        </div>
      </main>

      <Footer />
    </div>
  )
}
