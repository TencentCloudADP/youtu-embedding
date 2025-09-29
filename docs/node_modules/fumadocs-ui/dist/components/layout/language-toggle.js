'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useI18n } from '../../contexts/i18n.js';
import { Popover, PopoverContent, PopoverTrigger, } from '../../components/ui/popover.js';
import { cn } from '../../utils/cn.js';
import { buttonVariants } from '../../components/ui/button.js';
export function LanguageToggle(props) {
    const context = useI18n();
    if (!context.locales)
        throw new Error('Missing `<I18nProvider />`');
    return (_jsxs(Popover, { children: [_jsx(PopoverTrigger, { "aria-label": context.text.chooseLanguage, ...props, className: cn(buttonVariants({
                    color: 'ghost',
                    className: 'gap-1.5 p-1.5',
                }), props.className), children: props.children }), _jsxs(PopoverContent, { className: "flex flex-col overflow-hidden p-0", children: [_jsx("p", { className: "mb-1 p-2 text-xs font-medium text-fd-muted-foreground", children: context.text.chooseLanguage }), context.locales.map((item) => (_jsx("button", { type: "button", className: cn('p-2 text-start text-sm', item.locale === context.locale
                            ? 'bg-fd-primary/10 font-medium text-fd-primary'
                            : 'hover:bg-fd-accent hover:text-fd-accent-foreground'), onClick: () => {
                            context.onChange?.(item.locale);
                        }, children: item.name }, item.locale)))] })] }));
}
export function LanguageToggleText(props) {
    const context = useI18n();
    const text = context.locales?.find((item) => item.locale === context.locale)?.name;
    return _jsx("span", { ...props, children: text });
}
