'use client';
import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import * as Primitive from 'fumadocs-core/toc';
import { createContext, useContext, useRef } from 'react';
import { cn } from '../../utils/cn.js';
import { useI18n } from '../../contexts/i18n.js';
import { TocThumb } from '../../components/layout/toc-thumb.js';
import { mergeRefs } from '../../utils/merge-refs.js';
const TOCContext = createContext([]);
export function useTOCItems() {
    return useContext(TOCContext);
}
export function TOCProvider({ toc, children, ...props }) {
    return (_jsx(TOCContext, { value: toc, children: _jsx(Primitive.AnchorProvider, { toc: toc, ...props, children: children }) }));
}
export function TOCScrollArea({ ref, className, ...props }) {
    const viewRef = useRef(null);
    return (_jsx("div", { ref: mergeRefs(viewRef, ref), className: cn('relative min-h-0 text-sm ms-px overflow-auto [scrollbar-width:none] [mask-image:linear-gradient(to_bottom,transparent,white_16px,white_calc(100%-16px),transparent)] py-3', className), ...props, children: _jsx(Primitive.ScrollProvider, { containerRef: viewRef, children: props.children }) }));
}
export function TOCItems({ ref, className, ...props }) {
    const containerRef = useRef(null);
    const items = useTOCItems();
    const { text } = useI18n();
    if (items.length === 0)
        return (_jsx("div", { className: "rounded-lg border bg-fd-card p-3 text-xs text-fd-muted-foreground", children: text.tocNoHeadings }));
    return (_jsxs(_Fragment, { children: [_jsx(TocThumb, { containerRef: containerRef, className: "absolute top-(--fd-top) h-(--fd-height) w-px bg-fd-primary transition-all" }), _jsx("div", { ref: mergeRefs(ref, containerRef), className: cn('flex flex-col border-s border-fd-foreground/10', className), ...props, children: items.map((item) => (_jsx(TOCItem, { item: item }, item.url))) })] }));
}
function TOCItem({ item }) {
    return (_jsx(Primitive.TOCItem, { href: item.url, className: cn('prose py-1.5 text-sm text-fd-muted-foreground transition-colors [overflow-wrap:anywhere] first:pt-0 last:pb-0 data-[active=true]:text-fd-primary', item.depth <= 2 && 'ps-3', item.depth === 3 && 'ps-6', item.depth >= 4 && 'ps-8'), children: item.title }));
}
