import type { Metadata } from 'next';
import { Cormorant_Garamond, Noto_Sans_JP } from 'next/font/google';
import { ConditionalClerkProvider } from '@/components/providers/ConditionalClerkProvider';
import './globals.css';

const cormorantGaramond = Cormorant_Garamond({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-serif',
  display: 'swap',
});

const notoSansJP = Noto_Sans_JP({
  subsets: ['latin'],
  weight: ['300', '400', '500', '700'],
  variable: '--font-sans',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Cao - 理想の自分を、AIでシミュレーション',
  description:
    'あなたの顔写真と理想の顔を組み合わせて、段階的な変化をシミュレーション。美容医療の相談をより具体的に。',
  keywords: ['AI', '顔シミュレーション', '美容医療', 'モーフィング', '顔分析'],
  openGraph: {
    title: 'Cao - 理想の自分を、AIでシミュレーション',
    description:
      'あなたの顔写真と理想の顔を組み合わせて、段階的な変化をシミュレーション。',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ja" className={`${cormorantGaramond.variable} ${notoSansJP.variable}`}>
      <body className="font-sans antialiased">
        <ConditionalClerkProvider>{children}</ConditionalClerkProvider>
      </body>
    </html>
  );
}
