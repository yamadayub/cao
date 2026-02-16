import { getRequestConfig } from 'next-intl/server';
import { hasLocale } from 'next-intl';
import { routing } from './routing';

export default getRequestConfig(async ({ requestLocale }) => {
  const requested = await requestLocale;
  const locale = hasLocale(routing.locales, requested)
    ? requested
    : routing.defaultLocale;

  const [
    common,
    landing,
    simulate,
    result,
    mypage,
    modals,
    errors,
    parts,
    share,
    terms,
    privacy,
    auth,
    metadata,
  ] = await Promise.all([
    import(`./messages/${locale}/common.json`),
    import(`./messages/${locale}/landing.json`),
    import(`./messages/${locale}/simulate.json`),
    import(`./messages/${locale}/result.json`),
    import(`./messages/${locale}/mypage.json`),
    import(`./messages/${locale}/modals.json`),
    import(`./messages/${locale}/errors.json`),
    import(`./messages/${locale}/parts.json`),
    import(`./messages/${locale}/share.json`),
    import(`./messages/${locale}/terms.json`),
    import(`./messages/${locale}/privacy.json`),
    import(`./messages/${locale}/auth.json`),
    import(`./messages/${locale}/metadata.json`),
  ]);

  return {
    locale,
    messages: {
      common: common.default,
      landing: landing.default,
      simulate: simulate.default,
      result: result.default,
      mypage: mypage.default,
      modals: modals.default,
      errors: errors.default,
      parts: parts.default,
      share: share.default,
      terms: terms.default,
      privacy: privacy.default,
      auth: auth.default,
      metadata: metadata.default,
    },
  };
});
