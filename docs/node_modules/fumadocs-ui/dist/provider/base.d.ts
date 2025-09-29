import { ThemeProvider } from 'next-themes';
import { type ComponentPropsWithoutRef, type ReactNode } from 'react';
import type { DefaultSearchDialogProps } from '../components/dialog/search-default.js';
import { type SearchProviderProps } from '../contexts/search.js';
import { type LocaleItem, type Translations } from '../contexts/i18n.js';
interface SearchOptions extends Omit<SearchProviderProps, 'options' | 'children'> {
    options?: Partial<DefaultSearchDialogProps>;
    /**
     * Enable search functionality
     *
     * @defaultValue `true`
     */
    enabled?: boolean;
}
export interface RootProviderProps {
    /**
     * `dir` option for Radix UI
     */
    dir?: 'rtl' | 'ltr';
    /**
     * @remarks `SearchProviderProps`
     */
    search?: Partial<SearchOptions>;
    /**
     * Customise options of `next-themes`
     */
    theme?: Partial<ComponentPropsWithoutRef<typeof ThemeProvider>> & {
        /**
         * Enable `next-themes`
         *
         * @defaultValue true
         */
        enabled?: boolean;
    };
    i18n?: Omit<I18nProviderProps, 'children'>;
    children?: ReactNode;
}
export interface I18nProviderProps {
    /**
     * Current locale
     */
    locale: string;
    /**
     * Handle changes to the locale, redirect user when not specified.
     */
    onLocaleChange?: (v: string) => void;
    /**
     * Translations of current locale
     */
    translations?: Partial<Translations>;
    /**
     * Available languages
     */
    locales?: LocaleItem[];
    children?: ReactNode;
}
export declare function RootProvider({ children, dir, theme, search, i18n, }: RootProviderProps): import("react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=base.d.ts.map