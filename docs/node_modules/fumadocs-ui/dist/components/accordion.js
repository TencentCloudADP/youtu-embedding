'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import * as AccordionPrimitive from '@radix-ui/react-accordion';
import { Check, ChevronRight, Link as LinkIcon } from '../icons.js';
import { forwardRef, useEffect, useRef, useState, } from 'react';
import { cn } from '../utils/cn.js';
import { useCopyButton } from '../utils/use-copy-button.js';
import { buttonVariants } from '../components/ui/button.js';
import { mergeRefs } from '../utils/merge-refs.js';
export const Accordions = forwardRef(({ type = 'single', className, defaultValue, ...props }, ref) => {
    const rootRef = useRef(null);
    const composedRef = mergeRefs(ref, rootRef);
    const [value, setValue] = useState(() => type === 'single' ? (defaultValue ?? '') : (defaultValue ?? []));
    useEffect(() => {
        const id = window.location.hash.substring(1);
        const element = rootRef.current;
        if (!element || id.length === 0)
            return;
        const selected = document.getElementById(id);
        if (!selected || !element.contains(selected))
            return;
        const value = selected.getAttribute('data-accordion-value');
        if (value)
            setValue((prev) => (typeof prev === 'string' ? value : [value, ...prev]));
    }, []);
    return (
    // @ts-expect-error -- Multiple types
    _jsx(AccordionPrimitive.Root, { type: type, ref: composedRef, value: value, onValueChange: setValue, collapsible: type === 'single' ? true : undefined, className: cn('divide-y divide-fd-border overflow-hidden rounded-lg border bg-fd-card', className), ...props }));
});
Accordions.displayName = 'Accordions';
export const Accordion = forwardRef(({ title, className, id, value = String(title), children, ...props }, ref) => {
    return (_jsxs(AccordionPrimitive.Item, { ref: ref, value: value, className: cn('scroll-m-24', className), ...props, children: [_jsxs(AccordionPrimitive.Header, { id: id, "data-accordion-value": value, className: "not-prose flex flex-row items-center text-fd-card-foreground font-medium has-focus-visible:bg-fd-accent", children: [_jsxs(AccordionPrimitive.Trigger, { className: "group flex flex-1 items-center gap-2 px-3 py-2.5 text-start focus-visible:outline-none", children: [_jsx(ChevronRight, { className: "size-4 shrink-0 text-fd-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-90" }), title] }), id ? _jsx(CopyButton, { id: id }) : null] }), _jsx(AccordionPrimitive.Content, { className: "overflow-hidden data-[state=closed]:animate-fd-accordion-up data-[state=open]:animate-fd-accordion-down", children: _jsx("div", { className: "px-4 pb-2 text-[15px] prose-no-margin", children: children }) })] }));
});
function CopyButton({ id }) {
    const [checked, onClick] = useCopyButton(() => {
        const url = new URL(window.location.href);
        url.hash = id;
        return navigator.clipboard.writeText(url.toString());
    });
    return (_jsx("button", { type: "button", "aria-label": "Copy Link", className: cn(buttonVariants({
            color: 'ghost',
            className: 'text-fd-muted-foreground me-2',
        })), onClick: onClick, children: checked ? (_jsx(Check, { className: "size-3.5" })) : (_jsx(LinkIcon, { className: "size-3.5" })) }));
}
Accordion.displayName = 'Accordion';
