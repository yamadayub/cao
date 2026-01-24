import { redirect } from 'next/navigation';
import { MypageClient } from './MypageClient';

export const metadata = {
  title: 'マイページ - Cao',
  description: '保存済みシミュレーションの管理',
};

export default async function MyPage() {
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
