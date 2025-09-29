export interface Translations {
    search: string;
    searchNoResult: string;
    toc: string;
    tocNoHeadings: string;
    lastUpdate: string;
    chooseLanguage: string;
    nextPage: string;
    previousPage: string;
    chooseTheme: string;
    editOnGithub: string;
}
export interface LocaleItem {
    name: string;
    locale: string;
}
interface I18nContextType {
    locale?: string;
    onChange?: (v: string) => void;
    text: Translations;
    locales?: LocaleItem[];
}
export declare const defaultTranslations: Translations;
export declare const I18nContext: import("react").Context<I18nContextType>;
export declare function I18nLabel(props: {
    label: keyof Translations;
}): string;
export declare function useI18n(): I18nContextType;
export {};
//# sourceMappingURL=i18n.d.ts.map