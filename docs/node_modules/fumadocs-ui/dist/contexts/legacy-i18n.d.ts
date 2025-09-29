import type { I18nProviderProps } from '../provider/base.js';
/**
 * @deprecated legacy I18n Provider, use `<RootProvider i18n={...} />` instead
 */
export declare function I18nProvider({ locales, locale, onChange: _onChange, onLocaleChange, ...props }: I18nProviderProps & {
    /**
     * @deprecated use `onLocaleChange` instead
     */
    onChange?: I18nProviderProps['onLocaleChange'];
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=legacy-i18n.d.ts.map