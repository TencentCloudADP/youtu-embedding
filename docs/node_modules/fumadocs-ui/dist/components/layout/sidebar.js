'use client';
import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { ChevronDown, ExternalLink } from '../../icons.js';
import { usePathname } from 'fumadocs-core/framework';
import { createContext, Fragment, useContext, useMemo, useRef, useState, } from 'react';
import Link from 'fumadocs-core/link';
import { useOnChange } from 'fumadocs-core/utils/use-on-change';
import { cn } from '../../utils/cn.js';
import { ScrollArea, ScrollViewport } from '../../components/ui/scroll-area.js';
import { isActive } from '../../utils/is-active.js';
import { Collapsible, CollapsibleContent, CollapsibleTrigger, } from '../../components/ui/collapsible.js';
import { useSidebar } from '../../contexts/sidebar.js';
import { cva } from 'class-variance-authority';
import { useTreeContext, useTreePath } from '../../contexts/tree.js';
import { useMediaQuery } from 'fumadocs-core/utils/use-media-query';
import { Presence } from '@radix-ui/react-presence';
const itemVariants = cva('relative flex flex-row items-center gap-2 rounded-lg p-2 ps-(--sidebar-item-offset) text-start text-fd-muted-foreground [overflow-wrap:anywhere] [&_svg]:size-4 [&_svg]:shrink-0', {
    variants: {
        active: {
            true: 'bg-fd-primary/10 text-fd-primary',
            false: 'transition-colors hover:bg-fd-accent/50 hover:text-fd-accent-foreground/80 hover:transition-none',
        },
    },
});
const Context = createContext(null);
const FolderContext = createContext(null);
export function Sidebar({ defaultOpenLevel = 0, prefetch = true, Mobile, Content, }) {
    const isMobile = useMediaQuery('(width < 768px)') ?? false;
    const context = useMemo(() => {
        return {
            defaultOpenLevel,
            prefetch,
            level: 1,
        };
    }, [defaultOpenLevel, prefetch]);
    return (_jsx(Context.Provider, { value: context, children: isMobile && Mobile != null ? Mobile : Content }));
}
export function SidebarContent(props) {
    const { collapsed } = useSidebar();
    const [hover, setHover] = useState(false);
    const timerRef = useRef(0);
    const closeTimeRef = useRef(0);
    useOnChange(collapsed, () => {
        setHover(false);
        closeTimeRef.current = Date.now() + 150;
    });
    return (_jsx("aside", { id: "nd-sidebar", ...props, "data-collapsed": collapsed, className: cn('fixed left-0 rtl:left-auto rtl:right-(--removed-body-scroll-bar-size,0) flex flex-col items-end top-(--fd-sidebar-top) bottom-(--fd-sidebar-margin) z-20 bg-fd-card text-sm border-e transition-[top,opacity,translate,width] duration-200 max-md:hidden *:w-(--fd-sidebar-width)', collapsed && [
            'rounded-xl border translate-x-(--fd-sidebar-offset) rtl:-translate-x-(--fd-sidebar-offset)',
            hover ? 'z-50 shadow-lg' : 'opacity-0',
        ], props.className), style: {
            ...props.style,
            '--fd-sidebar-offset': hover
                ? 'calc(var(--spacing) * 2)'
                : 'calc(16px - 100%)',
            '--fd-sidebar-margin': collapsed ? '0.5rem' : '0px',
            '--fd-sidebar-top': `calc(var(--fd-banner-height) + var(--fd-nav-height) + var(--fd-sidebar-margin))`,
            width: collapsed
                ? 'var(--fd-sidebar-width)'
                : 'calc(var(--spacing) + var(--fd-sidebar-width) + var(--fd-layout-offset))',
        }, onPointerEnter: (e) => {
            if (!collapsed ||
                e.pointerType === 'touch' ||
                closeTimeRef.current > Date.now())
                return;
            window.clearTimeout(timerRef.current);
            setHover(true);
        }, onPointerLeave: (e) => {
            if (!collapsed || e.pointerType === 'touch')
                return;
            window.clearTimeout(timerRef.current);
            timerRef.current = window.setTimeout(() => {
                setHover(false);
                closeTimeRef.current = Date.now() + 150;
            }, Math.min(e.clientX, document.body.clientWidth - e.clientX) > 100
                ? 0
                : 500);
        }, children: props.children }));
}
export function SidebarContentMobile({ className, children, ...props }) {
    const { open, setOpen } = useSidebar();
    const state = open ? 'open' : 'closed';
    return (_jsxs(_Fragment, { children: [_jsx(Presence, { present: open, children: _jsx("div", { "data-state": state, className: "fixed z-40 inset-0 backdrop-blur-xs data-[state=open]:animate-fd-fade-in data-[state=closed]:animate-fd-fade-out", onClick: () => setOpen(false) }) }), _jsx(Presence, { present: open, children: ({ present }) => (_jsx("aside", { id: "nd-sidebar-mobile", ...props, "data-state": state, className: cn('fixed text-[15px] flex flex-col shadow-lg border-s end-0 inset-y-0 w-[85%] max-w-[380px] z-40 bg-fd-background data-[state=open]:animate-fd-sidebar-in data-[state=closed]:animate-fd-sidebar-out', !present && 'invisible', className), children: children })) })] }));
}
export function SidebarHeader(props) {
    return (_jsx("div", { ...props, className: cn('flex flex-col gap-3 p-4 pb-2', props.className), children: props.children }));
}
export function SidebarFooter(props) {
    return (_jsx("div", { ...props, className: cn('flex flex-col border-t p-4 pt-2', props.className), children: props.children }));
}
export function SidebarViewport(props) {
    return (_jsx(ScrollArea, { ...props, className: cn('h-full', props.className), children: _jsx(ScrollViewport, { className: "p-4 overscroll-contain", style: {
                '--sidebar-item-offset': 'calc(var(--spacing) * 2)',
                maskImage: 'linear-gradient(to bottom, transparent, white 12px, white calc(100% - 12px), transparent)',
            }, children: props.children }) }));
}
export function SidebarSeparator(props) {
    return (_jsx("p", { ...props, className: cn('inline-flex items-center gap-2 mb-1.5 px-2 ps-(--sidebar-item-offset) empty:mb-0 [&_svg]:size-4 [&_svg]:shrink-0', props.className), children: props.children }));
}
export function SidebarItem({ icon, ...props }) {
    const pathname = usePathname();
    const active = props.href !== undefined && isActive(props.href, pathname, false);
    const { prefetch } = useInternalContext();
    return (_jsxs(Link, { ...props, "data-active": active, className: cn(itemVariants({ active }), props.className), prefetch: prefetch, children: [icon ?? (props.external ? _jsx(ExternalLink, {}) : null), props.children] }));
}
export function SidebarFolder({ defaultOpen = false, ...props }) {
    const [open, setOpen] = useState(defaultOpen);
    useOnChange(defaultOpen, (v) => {
        if (v)
            setOpen(v);
    });
    return (_jsx(Collapsible, { open: open, onOpenChange: setOpen, ...props, children: _jsx(FolderContext.Provider, { value: useMemo(() => ({ open, setOpen }), [open]), children: props.children }) }));
}
export function SidebarFolderTrigger({ className, ...props }) {
    const { open } = useFolderContext();
    return (_jsxs(CollapsibleTrigger, { className: cn(itemVariants({ active: false }), 'w-full', className), ...props, children: [props.children, _jsx(ChevronDown, { "data-icon": true, className: cn('ms-auto transition-transform', !open && '-rotate-90') })] }));
}
export function SidebarFolderLink(props) {
    const { open, setOpen } = useFolderContext();
    const { prefetch } = useInternalContext();
    const pathname = usePathname();
    const active = props.href !== undefined && isActive(props.href, pathname, false);
    return (_jsxs(Link, { ...props, "data-active": active, className: cn(itemVariants({ active }), 'w-full', props.className), onClick: (e) => {
            if (e.target instanceof Element &&
                e.target.matches('[data-icon], [data-icon] *')) {
                setOpen(!open);
                e.preventDefault();
            }
            else {
                setOpen(active ? !open : true);
            }
        }, prefetch: prefetch, children: [props.children, _jsx(ChevronDown, { "data-icon": true, className: cn('ms-auto transition-transform', !open && '-rotate-90') })] }));
}
export function SidebarFolderContent(props) {
    const { level, ...ctx } = useInternalContext();
    return (_jsx(CollapsibleContent, { ...props, className: cn('relative', level === 1 && [
            "before:content-[''] before:absolute before:w-px before:inset-y-1 before:bg-fd-border before:start-2.5",
            "**:data-[active=true]:before:content-[''] **:data-[active=true]:before:bg-fd-primary **:data-[active=true]:before:absolute **:data-[active=true]:before:w-px **:data-[active=true]:before:inset-y-2.5 **:data-[active=true]:before:start-2.5",
        ], props.className), style: {
            '--sidebar-item-offset': `calc(var(--spacing) * ${(level + 1) * 3})`,
            ...props.style,
        }, children: _jsx(Context.Provider, { value: useMemo(() => ({
                ...ctx,
                level: level + 1,
            }), [ctx, level]), children: props.children }) }));
}
export function SidebarTrigger({ children, ...props }) {
    const { setOpen } = useSidebar();
    return (_jsx("button", { ...props, "aria-label": "Open Sidebar", onClick: () => setOpen((prev) => !prev), children: children }));
}
export function SidebarCollapseTrigger(props) {
    const { collapsed, setCollapsed } = useSidebar();
    return (_jsx("button", { type: "button", "aria-label": "Collapse Sidebar", "data-collapsed": collapsed, ...props, onClick: () => {
            setCollapsed((prev) => !prev);
        }, children: props.children }));
}
function useFolderContext() {
    const ctx = useContext(FolderContext);
    if (!ctx)
        throw new Error('Missing sidebar folder');
    return ctx;
}
function useInternalContext() {
    const ctx = useContext(Context);
    if (!ctx)
        throw new Error('<Sidebar /> component required.');
    return ctx;
}
/**
 * Render sidebar items from page tree
 */
export function SidebarPageTree(props) {
    const { root } = useTreeContext();
    return useMemo(() => {
        const { Separator, Item, Folder } = props.components ?? {};
        function renderSidebarList(items, level) {
            return items.map((item, i) => {
                if (item.type === 'separator') {
                    if (Separator)
                        return _jsx(Separator, { item: item }, i);
                    return (_jsxs(SidebarSeparator, { className: cn(i !== 0 && 'mt-6'), children: [item.icon, item.name] }, i));
                }
                if (item.type === 'folder') {
                    const children = renderSidebarList(item.children, level + 1);
                    if (Folder)
                        return (_jsx(Folder, { item: item, level: level, children: children }, i));
                    return (_jsx(PageTreeFolder, { item: item, children: children }, i));
                }
                if (Item)
                    return _jsx(Item, { item: item }, item.url);
                return (_jsx(SidebarItem, { href: item.url, external: item.external, icon: item.icon, children: item.name }, item.url));
            });
        }
        return (_jsx(Fragment, { children: renderSidebarList(root.children, 1) }, root.$id));
    }, [props.components, root]);
}
function PageTreeFolder({ item, ...props }) {
    const { defaultOpenLevel, level } = useInternalContext();
    const path = useTreePath();
    return (_jsxs(SidebarFolder, { defaultOpen: (item.defaultOpen ?? defaultOpenLevel >= level) || path.includes(item), children: [item.index ? (_jsxs(SidebarFolderLink, { href: item.index.url, external: item.index.external, ...props, children: [item.icon, item.name] })) : (_jsxs(SidebarFolderTrigger, { ...props, children: [item.icon, item.name] })), _jsx(SidebarFolderContent, { children: props.children })] }));
}
