import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { cn } from '../../utils/cn.js';
import { PageBreadcrumb, PageFooter, PageLastUpdate, PageTOC, PageTOCPopover, PageTOCPopoverContent, PageTOCPopoverTrigger, } from './page-client.js';
import { TOCItems, TOCProvider, TOCScrollArea } from '../../components/layout/toc.js';
import { Text } from '../../icons.js';
import { I18nLabel } from '../../contexts/i18n.js';
import ClerkTOCItems from '../../components/layout/toc-clerk.js';
export function PageTOCTitle(props) {
    return (_jsxs("h3", { ...props, className: cn('inline-flex items-center gap-1.5 text-sm text-fd-muted-foreground', props.className), children: [_jsx(Text, { className: "size-4" }), _jsx(I18nLabel, { label: "toc" })] }));
}
export function PageTOCItems({ variant = 'normal', ...props }) {
    return (_jsx(TOCScrollArea, { ...props, children: variant === 'clerk' ? _jsx(ClerkTOCItems, {}) : _jsx(TOCItems, {}) }));
}
export function PageTOCPopoverItems({ variant = 'normal', ...props }) {
    return (_jsx(TOCScrollArea, { ...props, children: variant === 'clerk' ? _jsx(ClerkTOCItems, {}) : _jsx(TOCItems, {}) }));
}
export function PageArticle(props) {
    return (_jsx("article", { ...props, className: cn('flex min-w-0 w-full flex-col gap-4 pt-8 px-4 md:px-6 md:mx-auto', props.className), children: props.children }));
}
export function PageRoot({ toc = false, children, ...props }) {
    const content = (_jsx("div", { id: "nd-page", ...props, className: cn('flex flex-1 w-full mx-auto max-w-(--fd-page-width) pt-(--fd-tocnav-height) pe-(--fd-toc-width)', props.className), children: children }));
    if (toc)
        return _jsx(TOCProvider, { ...toc, children: content });
    return content;
}
export { PageBreadcrumb, PageFooter, PageLastUpdate, PageTOC, PageTOCPopover, PageTOCPopoverTrigger, PageTOCPopoverContent, };
