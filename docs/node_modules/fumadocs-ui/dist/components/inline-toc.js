'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { ChevronDown } from '../icons.js';
import { Collapsible, CollapsibleContent, CollapsibleTrigger, } from './ui/collapsible.js';
import { cn } from '../utils/cn.js';
export function InlineTOC({ items, ...props }) {
    return (_jsxs(Collapsible, { ...props, className: cn('not-prose rounded-lg border bg-fd-card text-fd-card-foreground', props.className), children: [_jsxs(CollapsibleTrigger, { className: "group inline-flex w-full items-center justify-between px-4 py-2.5 font-medium", children: [props.children ?? 'Table of Contents', _jsx(ChevronDown, { className: "size-4 transition-transform duration-200 group-data-[state=open]:rotate-180" })] }), _jsx(CollapsibleContent, { children: _jsx("div", { className: "flex flex-col p-4 pt-0 text-sm text-fd-muted-foreground", children: items.map((item) => (_jsx("a", { href: item.url, className: "border-s py-1.5 hover:text-fd-accent-foreground", style: {
                            paddingInlineStart: 12 * Math.max(item.depth - 1, 0),
                        }, children: item.title }, item.url))) }) })] }));
}
