import type { Metadata } from 'next';
import { ConditionalClerkProvider } from '@/components/providers/ConditionalClerkProvider';
import './globals.css';

export const metadata: Metadata = {
  title: 'Cao - AI顔シミュレーション',
  description: 'AIを使った顔分析とモーフィングシミュレーション',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja">
      <body>
        <ConditionalClerkProvider>{children}</ConditionalClerkProvider>
      </body>
    </html>
  );
}
