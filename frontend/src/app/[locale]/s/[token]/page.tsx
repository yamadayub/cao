import { Metadata } from 'next';
import { setRequestLocale, getTranslations } from 'next-intl/server';
import { ShareViewClient } from './ShareViewClient';

interface SharePageProps {
  params: Promise<{
    locale: string;
    token: string;
  }>;
}

export async function generateMetadata({ params }: SharePageProps): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'share' });

  return {
    title: t('metadata.title'),
    description: t('metadata.description'),
  };
}

export default async function SharePage({ params }: SharePageProps) {
  const { locale, token } = await params;
  setRequestLocale(locale);

  return <ShareViewClient token={token} testId="share-view" />;
}
