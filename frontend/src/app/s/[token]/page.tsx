import { Metadata } from 'next';
import { ShareViewClient } from './ShareViewClient';

interface SharePageProps {
  params: Promise<{
    token: string;
  }>;
}

export async function generateMetadata({ params }: SharePageProps): Promise<Metadata> {
  return {
    title: '共有シミュレーション - Cao',
    description: '共有されたAI顔シミュレーション結果を確認',
  };
}

export default async function SharePage({ params }: SharePageProps) {
  const { token } = await params;

  return <ShareViewClient token={token} testId="share-view" />;
}
