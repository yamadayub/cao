import { NextIntlClientProvider, hasLocale } from 'next-intl';
import { setRequestLocale, getMessages, getTranslations } from 'next-intl/server';
import { notFound } from 'next/navigation';
import { Cormorant_Garamond, Noto_Sans_JP, Noto_Sans_SC, Noto_Sans_TC, Noto_Sans_KR } from 'next/font/google';
import Script from 'next/script';
import type { Metadata, Viewport } from 'next';
import { ConditionalClerkProvider } from '@/components/providers/ConditionalClerkProvider';
import { routing } from '@/i18n/routing';
import { locales, defaultLocale, type Locale } from '@/i18n/config';
import '../globals.css';

const cormorantGaramond = Cormorant_Garamond({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-serif',
  display: 'swap',
});

const notoSansJP = Noto_Sans_JP({
  subsets: ['latin'],
  weight: ['300', '400', '500', '700'],
  variable: '--font-sans-ja',
  display: 'swap',
});

const notoSansSC = Noto_Sans_SC({
  subsets: ['latin'],
  weight: ['300', '400', '500', '700'],
  variable: '--font-sans-zh-cn',
  display: 'swap',
});

const notoSansTC = Noto_Sans_TC({
  subsets: ['latin'],
  weight: ['300', '400', '500', '700'],
  variable: '--font-sans-zh-tw',
  display: 'swap',
});

const notoSansKR = Noto_Sans_KR({
  subsets: ['latin'],
  weight: ['300', '400', '500', '700'],
  variable: '--font-sans-ko',
  display: 'swap',
});

const fontSansVariableMap: Record<Locale, string> = {
  ja: notoSansJP.variable,
  en: notoSansJP.variable,
  'zh-CN': notoSansSC.variable,
  'zh-TW': notoSansTC.variable,
  ko: notoSansKR.variable,
};

const fontSansClassMap: Record<Locale, string> = {
  ja: '--font-sans-ja',
  en: '--font-sans-ja',
  'zh-CN': '--font-sans-zh-cn',
  'zh-TW': '--font-sans-zh-tw',
  ko: '--font-sans-ko',
};

const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || 'https://cao-ai.com';
const GA_ID = process.env.NEXT_PUBLIC_GA_ID;

function getAlternateLanguages(path: string = '') {
  const languages: Record<string, string> = {};
  for (const locale of locales) {
    const prefix = locale === defaultLocale ? '' : `/${locale}`;
    languages[locale] = `${BASE_URL}${prefix}${path}`;
  }
  languages['x-default'] = `${BASE_URL}${path}`;
  return languages;
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

export async function generateMetadata({ params }: { params: Promise<{ locale: string }> }): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'metadata' });

  return {
    metadataBase: new URL(BASE_URL),
    title: {
      default: t('home.title'),
      template: '%s',
    },
    description: t('home.description'),
    keywords: t('keywords'),
    icons: {
      icon: '/icon.svg',
    },
    openGraph: {
      title: t('home.ogTitle'),
      description: t('home.ogDescription'),
      siteName: 'Cao',
      locale,
      type: 'website',
    },
    alternates: {
      languages: getAlternateLanguages(),
    },
  };
}

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

type Props = {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
};

export default async function LocaleLayout({ children, params }: Props) {
  const { locale } = await params;

  if (!hasLocale(routing.locales, locale)) {
    notFound();
  }

  setRequestLocale(locale);

  const messages = await getMessages();

  const fontVars = [
    cormorantGaramond.variable,
    notoSansJP.variable,
    notoSansSC.variable,
    notoSansTC.variable,
    notoSansKR.variable,
  ].join(' ');

  const activeSansVar = fontSansClassMap[locale as Locale] || '--font-sans-ja';

  return (
    <html lang={locale} className={fontVars}>
      {GA_ID && (
        <head>
          <Script
            src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}
            strategy="afterInteractive"
          />
          <Script id="google-analytics" strategy="afterInteractive">
            {`window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_ID}');`}
          </Script>
        </head>
      )}
      <body
        className="font-sans antialiased"
        style={{ fontFamily: `var(${activeSansVar}), sans-serif` }}
      >
        <ConditionalClerkProvider locale={locale}>
          <NextIntlClientProvider messages={messages}>
            {children}
          </NextIntlClientProvider>
        </ConditionalClerkProvider>
      </body>
    </html>
  );
}
