import { redirect } from 'next/navigation';
import type { Metadata } from 'next';
import { setRequestLocale, getTranslations } from 'next-intl/server';
import { MypageClient } from './MypageClient';

type Props = {
  params: Promise<{ locale: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'metadata' });

  return {
    title: t('mypage.title'),
    description: t('mypage.description'),
  };
}

export default async function MyPage({ params }: Props) {
  const { locale } = await params;
  setRequestLocale(locale);

  // Clerkキーが設定されていない場合は/sign-inにリダイレクト
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    redirect('/sign-in');
  }

  // 動的インポートでClerkが設定されている場合のみ読み込む
  const { currentUser } = await import('@clerk/nextjs/server');
  const user = await currentUser();

  if (!user) {
    redirect('/sign-in');
  }

  return <MypageClient testId="mypage" />;
}
