import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { forwardRef } from 'react';
import { cn } from './utils/cn.js';
import { buttonVariants } from './components/ui/button.js';
import { Edit } from './icons.js';
import { I18nLabel } from './contexts/i18n.js';
import { PageArticle, PageBreadcrumb, PageFooter, PageLastUpdate, PageRoot, PageTOC, PageTOCItems, PageTOCPopover, PageTOCPopoverContent, PageTOCPopoverItems, PageTOCPopoverTrigger, PageTOCTitle, } from './layouts/docs/page.js';
export function DocsPage({ editOnGithub, breadcrumb: { enabled: breadcrumbEnabled = true, component: breadcrumb, ...breadcrumbProps } = {}, footer = {}, lastUpdate, container, full = false, tableOfContentPopover: { enabled: tocPopoverEnabled, component: tocPopover, ...tocPopoverOptions } = {}, tableOfContent: { enabled: tocEnabled, component: tocReplace, ...tocOptions } = {}, toc = [], article, children, }) {
    // disable TOC on full mode, you can still enable it with `enabled` option.
    tocEnabled ?? (tocEnabled = !full &&
        (toc.length > 0 ||
            tocOptions.footer !== undefined ||
            tocOptions.header !== undefined));
    tocPopoverEnabled ?? (tocPopoverEnabled = toc.length > 0 ||
        tocPopoverOptions.header !== undefined ||
        tocPopoverOptions.footer !== undefined);
    return (_jsxs(PageRoot, { toc: tocEnabled || tocPopoverEnabled
            ? {
                toc,
                single: tocOptions.single,
            }
            : false, ...container, className: cn(!tocEnabled && '[--fd-toc-width:0px]', container?.className), children: [tocPopoverEnabled &&
                (tocPopover ?? (_jsxs(PageTOCPopover, { children: [_jsx(PageTOCPopoverTrigger, {}), _jsxs(PageTOCPopoverContent, { children: [tocPopoverOptions.header, _jsx(PageTOCPopoverItems, { variant: tocPopoverOptions.style }), tocPopoverOptions.footer] })] }))), _jsxs(PageArticle, { ...article, children: [breadcrumbEnabled &&
                        (breadcrumb ?? _jsx(PageBreadcrumb, { ...breadcrumbProps })), children, _jsxs("div", { className: "flex flex-row flex-wrap items-center justify-between gap-4 empty:hidden", children: [editOnGithub && (_jsx(EditOnGitHub, { href: `https://github.com/${editOnGithub.owner}/${editOnGithub.repo}/blob/${editOnGithub.sha}/${editOnGithub.path.startsWith('/') ? editOnGithub.path.slice(1) : editOnGithub.path}` })), lastUpdate && _jsx(PageLastUpdate, { date: new Date(lastUpdate) })] }), footer.enabled !== false &&
                        (footer.component ?? _jsx(PageFooter, { items: footer.items }))] }), tocEnabled &&
                (tocReplace ?? (_jsxs(PageTOC, { children: [tocOptions.header, _jsx(PageTOCTitle, {}), _jsx(PageTOCItems, { variant: tocOptions.style }), tocOptions.footer] })))] }));
}
export function EditOnGitHub(props) {
    return (_jsx("a", { target: "_blank", rel: "noreferrer noopener", ...props, className: cn(buttonVariants({
            color: 'secondary',
            size: 'sm',
            className: 'gap-1.5 not-prose',
        }), props.className), children: props.children ?? (_jsxs(_Fragment, { children: [_jsx(Edit, { className: "size-3.5" }), _jsx(I18nLabel, { label: "editOnGithub" })] })) }));
}
/**
 * Add typography styles
 */
export const DocsBody = forwardRef((props, ref) => (_jsx("div", { ref: ref, ...props, className: cn('prose flex-1', props.className), children: props.children })));
DocsBody.displayName = 'DocsBody';
export const DocsDescription = forwardRef((props, ref) => {
    // don't render if no description provided
    if (props.children === undefined)
        return null;
    return (_jsx("p", { ref: ref, ...props, className: cn('mb-8 text-lg text-fd-muted-foreground', props.className), children: props.children }));
});
DocsDescription.displayName = 'DocsDescription';
export const DocsTitle = forwardRef((props, ref) => {
    return (_jsx("h1", { ref: ref, ...props, className: cn('text-[1.75em] font-semibold', props.className), children: props.children }));
});
DocsTitle.displayName = 'DocsTitle';
/**
 * For separate MDX page
 */
export function withArticle(props) {
    return (_jsx("main", { ...props, className: cn('container py-12', props.className), children: _jsx("article", { className: "prose", children: props.children }) }));
}
