import type { MetadataRoute } from 'next';
import { locales, defaultLocale } from '@/i18n/config';

const BASE_URL = 'https://cao.style-elements.jp';

const staticPages = [
  { path: '/', changeFrequency: 'weekly' as const, priority: 1.0 },
  { path: '/simulate', changeFrequency: 'weekly' as const, priority: 0.9 },
  { path: '/terms', changeFrequency: 'monthly' as const, priority: 0.3 },
  { path: '/privacy', changeFrequency: 'monthly' as const, priority: 0.3 },
  { path: '/sign-in', changeFrequency: 'monthly' as const, priority: 0.5 },
  { path: '/sign-up', changeFrequency: 'monthly' as const, priority: 0.5 },
];

export default function sitemap(): MetadataRoute.Sitemap {
  const entries: MetadataRoute.Sitemap = [];

  for (const page of staticPages) {
    for (const locale of locales) {
      const prefix = locale === defaultLocale ? '' : `/${locale}`;
      entries.push({
        url: `${BASE_URL}${prefix}${page.path === '/' ? '' : page.path}`,
        lastModified: new Date(),
        changeFrequency: page.changeFrequency,
        priority: page.priority,
      });
    }
  }

  return entries;
}
