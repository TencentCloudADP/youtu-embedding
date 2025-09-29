'use client';
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { Check, ChevronsUpDown } from '../../icons.js';
import { useMemo, useState } from 'react';
import Link from 'fumadocs-core/link';
import { usePathname } from 'fumadocs-core/framework';
import { cn } from '../../utils/cn.js';
import { isTabActive } from '../../utils/is-active.js';
import { useSidebar } from '../../contexts/sidebar.js';
import { Popover, PopoverContent, PopoverTrigger } from '../ui/popover.js';
export function RootToggle({ options, placeholder, ...props }) {
    const [open, setOpen] = useState(false);
    const { closeOnRedirect } = useSidebar();
    const pathname = usePathname();
    const selected = useMemo(() => {
        return options.findLast((item) => isTabActive(item, pathname));
    }, [options, pathname]);
    const onClick = () => {
        closeOnRedirect.current = false;
        setOpen(false);
    };
    const item = selected ? (_jsxs(_Fragment, { children: [_jsx("div", { className: "size-9 shrink-0 md:size-5", children: selected.icon }), _jsxs("div", { children: [_jsx("p", { className: "text-sm font-medium", children: selected.title }), _jsx("p", { className: "text-[13px] text-fd-muted-foreground empty:hidden md:hidden", children: selected.description })] })] })) : (placeholder);
    return (_jsxs(Popover, { open: open, onOpenChange: setOpen, children: [item && (_jsxs(PopoverTrigger, { ...props, className: cn('flex items-center gap-2 rounded-lg p-2 border bg-fd-secondary/50 text-start text-fd-secondary-foreground transition-colors hover:bg-fd-accent data-[state=open]:bg-fd-accent data-[state=open]:text-fd-accent-foreground', props.className), children: [item, _jsx(ChevronsUpDown, { className: "shrink-0 ms-auto size-4 text-fd-muted-foreground" })] })), _jsx(PopoverContent, { className: "flex flex-col gap-1 w-(--radix-popover-trigger-width) overflow-hidden p-1", children: options.map((item) => {
                    const isActive = selected && item.url === selected.url;
                    if (!isActive && item.unlisted)
                        return;
                    return (_jsxs(Link, { href: item.url, onClick: onClick, ...item.props, className: cn('flex items-center gap-2 rounded-lg p-1.5 hover:bg-fd-accent hover:text-fd-accent-foreground', item.props?.className), children: [_jsx("div", { className: "shrink-0 size-9 md:mt-1 md:mb-auto md:size-5", children: item.icon }), _jsxs("div", { children: [_jsx("p", { className: "text-sm font-medium", children: item.title }), _jsx("p", { className: "text-[13px] text-fd-muted-foreground empty:hidden", children: item.description })] }), _jsx(Check, { className: cn('shrink-0 ms-auto size-3.5 text-fd-primary', !isActive && 'invisible') })] }, item.url));
                }) })] }));
}
