'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import * as React from 'react';
import { createContext, useContext, useEffect, useId, useMemo, useState, } from 'react';
import { cn } from '../utils/cn.js';
import * as Unstyled from './tabs.unstyled.js';
const TabsContext = createContext(null);
function useTabContext() {
    const ctx = useContext(TabsContext);
    if (!ctx)
        throw new Error('You must wrap your component in <Tabs>');
    return ctx;
}
export const TabsList = React.forwardRef((props, ref) => (_jsx(Unstyled.TabsList, { ref: ref, ...props, className: cn('flex gap-3.5 text-fd-secondary-foreground overflow-x-auto px-4 not-prose', props.className) })));
TabsList.displayName = 'TabsList';
export const TabsTrigger = React.forwardRef((props, ref) => (_jsx(Unstyled.TabsTrigger, { ref: ref, ...props, className: cn('inline-flex items-center gap-2 whitespace-nowrap text-fd-muted-foreground border-b border-transparent py-2 text-sm font-medium transition-colors [&_svg]:size-4 hover:text-fd-accent-foreground disabled:pointer-events-none disabled:opacity-50 data-[state=active]:border-fd-primary data-[state=active]:text-fd-primary', props.className) })));
TabsTrigger.displayName = 'TabsTrigger';
export function Tabs({ ref, className, items, label, defaultIndex = 0, defaultValue = items ? escapeValue(items[defaultIndex]) : undefined, ...props }) {
    const [value, setValue] = useState(defaultValue);
    const collection = useMemo(() => [], []);
    return (_jsxs(Unstyled.Tabs, { ref: ref, className: cn('flex flex-col overflow-hidden rounded-xl border bg-fd-secondary my-4', className), value: value, onValueChange: (v) => {
            if (items && !items.some((item) => escapeValue(item) === v))
                return;
            setValue(v);
        }, ...props, children: [items && (_jsxs(TabsList, { children: [label && (_jsx("span", { className: "text-sm font-medium my-auto me-auto", children: label })), items.map((item) => (_jsx(TabsTrigger, { value: escapeValue(item), children: item }, item)))] })), _jsx(TabsContext.Provider, { value: useMemo(() => ({ items, collection }), [collection, items]), children: props.children })] }));
}
export function Tab({ value, ...props }) {
    const { items } = useTabContext();
    const resolved = value ??
        // eslint-disable-next-line react-hooks/rules-of-hooks -- `value` is not supposed to change
        items?.at(useCollectionIndex());
    if (!resolved)
        throw new Error('Failed to resolve tab `value`, please pass a `value` prop to the Tab component.');
    return (_jsx(TabsContent, { value: escapeValue(resolved), ...props, children: props.children }));
}
export function TabsContent({ value, className, ...props }) {
    return (_jsx(Unstyled.TabsContent, { value: value, forceMount: true, className: cn('p-4 text-[15px] bg-fd-background rounded-xl outline-none prose-no-margin data-[state=inactive]:hidden [&>figure:only-child]:-m-4 [&>figure:only-child]:border-none', className), ...props, children: props.children }));
}
/**
 * Inspired by Headless UI.
 *
 * Return the index of children, this is made possible by registering the order of render from children using React context.
 * This is supposed by work with pre-rendering & pure client-side rendering.
 */
function useCollectionIndex() {
    const key = useId();
    const { collection } = useTabContext();
    useEffect(() => {
        return () => {
            const idx = collection.indexOf(key);
            if (idx !== -1)
                collection.splice(idx, 1);
        };
    }, [key, collection]);
    if (!collection.includes(key))
        collection.push(key);
    return collection.indexOf(key);
}
/**
 * only escape whitespaces in values in simple mode
 */
function escapeValue(v) {
    return v.toLowerCase().replace(/\s/, '-');
}
