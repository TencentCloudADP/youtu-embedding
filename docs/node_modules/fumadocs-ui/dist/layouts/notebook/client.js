'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { cn } from '../../utils/cn.js';
import { useMemo } from 'react';
import { useSidebar } from '../../contexts/sidebar.js';
import { useNav } from '../../contexts/layout.js';
import { buttonVariants } from '../../components/ui/button.js';
import { Sidebar as SidebarIcon } from '../../icons.js';
import Link from 'fumadocs-core/link';
import { usePathname } from 'fumadocs-core/framework';
import { isTabActive } from '../../utils/is-active.js';
export function Navbar({ mode, ...props }) {
    const { open, collapsed } = useSidebar();
    const { isTransparent } = useNav();
    return (_jsx("header", { id: "nd-subnav", ...props, className: cn('fixed flex flex-col top-(--fd-banner-height) left-0 right-(--removed-body-scroll-bar-size,0) z-10 px-(--fd-layout-offset) h-(--fd-nav-height) backdrop-blur-sm transition-colors', (!isTransparent || open) && 'bg-fd-background/80', mode === 'auto' &&
            !collapsed &&
            'ps-[calc(var(--fd-layout-offset)+var(--fd-sidebar-width))]', props.className), children: props.children }));
}
export function LayoutBody(props) {
    const { collapsed } = useSidebar();
    return (_jsx("main", { id: "nd-docs-layout", ...props, className: cn('flex flex-1 flex-col transition-[padding] pt-(--fd-nav-height) fd-notebook-layout', !collapsed && 'mx-(--fd-layout-offset)', props.className), style: {
            ...props.style,
            paddingInlineStart: collapsed
                ? 'min(calc(100vw - var(--fd-page-width)), var(--fd-sidebar-width))'
                : 'var(--fd-sidebar-width)',
        }, children: props.children }));
}
export function NavbarSidebarTrigger({ className, ...props }) {
    const { setOpen } = useSidebar();
    return (_jsx("button", { ...props, className: cn(buttonVariants({
            color: 'ghost',
            size: 'icon-sm',
            className,
        })), onClick: () => setOpen((prev) => !prev), children: _jsx(SidebarIcon, {}) }));
}
export function LayoutTabs({ options, ...props }) {
    const pathname = usePathname();
    const selected = useMemo(() => {
        return options.findLast((option) => isTabActive(option, pathname));
    }, [options, pathname]);
    return (_jsx("div", { ...props, className: cn('flex flex-row items-end gap-6 overflow-auto', props.className), children: options.map((option) => (_jsx(LayoutTab, { selected: selected === option, option: option }, option.url))) }));
}
function LayoutTab({ option: { title, url, unlisted, props }, selected = false, }) {
    return (_jsx(Link, { href: url, ...props, className: cn('inline-flex border-b-2 border-transparent transition-colors items-center pb-1.5 font-medium gap-2 text-fd-muted-foreground text-sm text-nowrap hover:text-fd-accent-foreground', unlisted && !selected && 'hidden', selected && 'border-fd-primary text-fd-primary', props?.className), children: title }));
}
