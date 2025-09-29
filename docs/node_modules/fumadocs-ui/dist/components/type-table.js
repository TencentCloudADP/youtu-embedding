'use client';
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { ChevronDown } from '../icons.js';
import Link from 'fumadocs-core/link';
import { cva } from 'class-variance-authority';
import { cn } from '../utils/cn.js';
import { useState } from 'react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger, } from '../components/ui/collapsible.js';
const keyVariants = cva('text-fd-primary', {
    variants: {
        deprecated: {
            true: 'line-through text-fd-primary/50',
        },
    },
});
const fieldVariants = cva('text-fd-muted-foreground not-prose pe-2');
export function TypeTable({ type }) {
    return (_jsxs("div", { className: "@container flex flex-col p-1 bg-fd-card text-fd-card-foreground rounded-2xl border my-6 text-sm overflow-hidden", children: [_jsxs("div", { className: "flex font-medium items-center px-3 py-1 not-prose text-fd-muted-foreground", children: [_jsx("p", { className: "w-[25%]", children: "Prop" }), _jsx("p", { className: "@max-xl:hidden", children: "Type" })] }), Object.entries(type).map(([key, value]) => (_jsx(Item, { name: key, item: value }, key)))] }));
}
function Item({ name, item: { parameters = [], description, required = false, deprecated, typeDescription, default: defaultValue, type, typeDescriptionLink, returns, }, }) {
    const [open, setOpen] = useState(false);
    return (_jsxs(Collapsible, { open: open, onOpenChange: setOpen, className: cn('rounded-xl border overflow-hidden transition-all', open
            ? 'shadow-sm bg-fd-background not-last:mb-2'
            : 'border-transparent'), children: [_jsxs(CollapsibleTrigger, { className: "relative flex flex-row items-center w-full group text-start px-3 py-2 not-prose hover:bg-fd-accent", children: [_jsxs("code", { className: cn(keyVariants({
                            deprecated,
                            className: 'min-w-fit w-[25%] font-medium',
                        })), children: [name, !required && '?'] }), typeDescriptionLink ? (_jsx(Link, { href: typeDescriptionLink, className: "underline @max-xl:hidden", children: type })) : (_jsx("span", { className: "@max-xl:hidden", children: type })), _jsx(ChevronDown, { className: "absolute end-2 size-4 text-fd-muted-foreground transition-transform group-data-[state=open]:rotate-180" })] }), _jsx(CollapsibleContent, { children: _jsxs("div", { className: "grid grid-cols-[1fr_3fr] gap-y-4 text-sm p-3 overflow-auto fd-scroll-container border-t", children: [_jsx("div", { className: "text-sm prose col-span-full prose-no-margin empty:hidden", children: description }), typeDescription && (_jsxs(_Fragment, { children: [_jsx("p", { className: cn(fieldVariants()), children: "Type" }), _jsx("p", { className: "my-auto not-prose", children: typeDescription })] })), defaultValue && (_jsxs(_Fragment, { children: [_jsx("p", { className: cn(fieldVariants()), children: "Default" }), _jsx("p", { className: "my-auto not-prose", children: defaultValue })] })), parameters.length > 0 && (_jsxs(_Fragment, { children: [_jsx("p", { className: cn(fieldVariants()), children: "Parameters" }), _jsx("div", { className: "flex flex-col gap-2", children: parameters.map((param) => (_jsxs("div", { className: "inline-flex items-center flex-wrap gap-1", children: [_jsxs("p", { className: "font-medium not-prose text-nowrap", children: [param.name, " -"] }), _jsx("div", { className: "text-sm prose prose-no-margin", children: param.description })] }, param.name))) })] })), returns && (_jsxs(_Fragment, { children: [_jsx("p", { className: cn(fieldVariants()), children: "Returns" }), _jsx("div", { className: "my-auto text-sm prose prose-no-margin", children: returns })] }))] }) })] }));
}
