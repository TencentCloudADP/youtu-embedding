export { defaultTranslations } from './contexts/i18n.js';
export { I18nProvider } from './contexts/legacy-i18n.js';
export function defineI18nUI(config, options) {
    const { translations } = options;
    return {
        provider(locale = config.defaultLanguage) {
            return {
                locale,
                translations: translations[locale],
                locales: config.languages.map((locale) => ({
                    locale,
                    name: translations[locale]?.displayName ?? locale,
                })),
            };
        },
    };
}
