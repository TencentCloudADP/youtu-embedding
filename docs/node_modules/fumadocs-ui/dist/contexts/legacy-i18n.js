'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { usePathname, useRouter } from 'fumadocs-core/framework';
import { useEffectEvent } from 'fumadocs-core/utils/use-effect-event';
import { useMemo } from 'react';
import { defaultTranslations, I18nContext } from '../contexts/i18n.js';
/**
 * @deprecated legacy I18n Provider, use `<RootProvider i18n={...} />` instead
 */
export function I18nProvider({ locales = [], locale, onChange: _onChange, onLocaleChange = _onChange, ...props }) {
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
        router.refresh();
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
