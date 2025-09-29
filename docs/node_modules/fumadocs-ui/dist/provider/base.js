'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { ThemeProvider } from 'next-themes';
import { lazy, useMemo, } from 'react';
import { DirectionProvider } from '@radix-ui/react-direction';
import { SidebarProvider } from '../contexts/sidebar.js';
import { SearchProvider } from '../contexts/search.js';
import { useEffectEvent } from 'fumadocs-core/utils/use-effect-event';
import { defaultTranslations, I18nContext, } from '../contexts/i18n.js';
import { usePathname, useRouter } from 'fumadocs-core/framework';
const DefaultSearchDialog = lazy(() => import('../components/dialog/search-default.js'));
export function RootProvider({ children, dir = 'ltr', theme = {}, search, i18n, }) {
    let body = children;
    if (search?.enabled !== false)
        body = (_jsx(SearchProvider, { SearchDialog: DefaultSearchDialog, ...search, children: body }));
    if (theme?.enabled !== false)
        body = (_jsx(ThemeProvider, { attribute: "class", defaultTheme: "system", enableSystem: true, disableTransitionOnChange: true, ...theme, children: body }));
    if (i18n) {
        body = _jsx(I18nProvider, { ...i18n, children: body });
    }
    return (_jsx(DirectionProvider, { dir: dir, children: _jsx(SidebarProvider, { children: body }) }));
}
function I18nProvider({ locales = [], locale, onLocaleChange, ...props }) {
    const router = useRouter();
    const pathname = usePathname();
    const onChange = useEffectEvent((value) => {
        if (onLocaleChange) {
            return onLocaleChange(value);
        }
        const segments = pathname.split('/').filter((v) => v.length > 0);
        // If locale prefix hidden
        if (segments[0] !== locale) {
            segments.unshift(value);
        }
        else {
            segments[0] = value;
        }
        router.push(`/${segments.join('/')}`);
    });
    return (_jsx(I18nContext.Provider, { value: useMemo(() => ({
            locale,
            locales,
            text: {
                ...defaultTranslations,
                ...props.translations,
            },
            onChange,
        }), [locale, locales, onChange, props.translations]), children: props.children }));
}
