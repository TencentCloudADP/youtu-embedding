'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Sidebar as SidebarIcon } from '../../icons.js';
import { cn } from '../../utils/cn.js';
import { buttonVariants } from '../../components/ui/button.js';
import { useSidebar } from '../../contexts/sidebar.js';
import { useNav } from '../../contexts/layout.js';
import { SidebarCollapseTrigger } from '../../components/layout/sidebar.js';
import { SearchToggle } from '../../components/layout/search-toggle.js';
export function Navbar(props) {
    const { isTransparent } = useNav();
    return (_jsx("header", { id: "nd-subnav", ...props, className: cn('fixed top-(--fd-banner-height) left-0 right-(--removed-body-scroll-bar-size,0) z-30 flex items-center ps-4 pe-2.5 border-b transition-colors backdrop-blur-sm', !isTransparent && 'bg-fd-background/80', props.className), children: props.children }));
}
export function LayoutBody(props) {
    const { collapsed } = useSidebar();
    return (_jsx("main", { id: "nd-docs-layout", ...props, className: cn('flex flex-1 flex-col pt-(--fd-nav-height) transition-[padding] fd-default-layout', !collapsed && 'mx-(--fd-layout-offset)', props.className), style: {
            ...props.style,
            paddingInlineStart: collapsed
                ? 'min(calc(100vw - var(--fd-page-width)), var(--fd-sidebar-width))'
                : 'var(--fd-sidebar-width)',
        }, children: props.children }));
}
export function CollapsibleControl() {
    const { collapsed } = useSidebar();
    return (_jsxs("div", { className: cn('fixed flex shadow-lg transition-opacity rounded-xl p-0.5 border bg-fd-muted text-fd-muted-foreground z-10 max-md:hidden xl:start-4 max-xl:end-4', !collapsed && 'pointer-events-none opacity-0'), style: {
            top: 'calc(var(--fd-banner-height) + var(--fd-tocnav-height) + var(--spacing) * 4)',
        }, children: [_jsx(SidebarCollapseTrigger, { className: cn(buttonVariants({
                    color: 'ghost',
                    size: 'icon-sm',
                    className: 'rounded-lg',
                })), children: _jsx(SidebarIcon, {}) }), _jsx(SearchToggle, { className: "rounded-lg", hideIfDisabled: true })] }));
}
