import type { Translations } from './contexts/i18n.js';
import type { I18nProviderProps } from './provider/base.js';
import type { I18nConfig } from 'fumadocs-core/i18n';
export type { I18nProviderProps, Translations };
export { defaultTranslations } from './contexts/i18n.js';
export { I18nProvider } from './contexts/legacy-i18n.js';
export declare function defineI18nUI<Languages extends string>(config: I18nConfig<Languages>, options: {
    translations: {
        [K in Languages]?: Partial<Translations> & {
            displayName?: string;
        };
    };
}): {
    provider(locale?: string): I18nProviderProps;
};
//# sourceMappingURL=i18n.d.ts.map