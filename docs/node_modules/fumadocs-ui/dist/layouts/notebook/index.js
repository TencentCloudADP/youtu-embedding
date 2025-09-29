import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { Fragment, useMemo, } from 'react';
import { BaseLinkItem, getLinks, } from '../../layouts/shared/index.js';
import { Sidebar, SidebarCollapseTrigger, SidebarContent, SidebarContentMobile, SidebarFolder, SidebarFolderContent, SidebarFolderLink, SidebarFolderTrigger, SidebarFooter, SidebarHeader, SidebarItem, SidebarPageTree, SidebarTrigger, SidebarViewport, } from '../../components/layout/sidebar.js';
import { TreeContextProvider } from '../../contexts/tree.js';
import { cn } from '../../utils/cn.js';
import { buttonVariants } from '../../components/ui/button.js';
import { ChevronDown, Languages, Sidebar as SidebarIcon, X, } from '../../icons.js';
import { LanguageToggle } from '../../components/layout/language-toggle.js';
import { ThemeToggle } from '../../components/layout/theme-toggle.js';
import { Popover, PopoverContent, PopoverTrigger, } from '../../components/ui/popover.js';
import { LayoutBody, LayoutTabs, Navbar, NavbarSidebarTrigger, } from '../../layouts/notebook/client.js';
import { NavProvider } from '../../contexts/layout.js';
import { RootToggle } from '../../components/layout/root-toggle.js';
import Link from 'fumadocs-core/link';
import { LargeSearchToggle, SearchToggle, } from '../../components/layout/search-toggle.js';
import { HideIfEmpty } from 'fumadocs-core/hide-if-empty';
import { getSidebarTabs, } from '../../utils/get-sidebar-tabs.js';
export function DocsLayout(props) {
    const { tabMode = 'sidebar', nav: { transparentMode, ...nav } = {}, sidebar: { tabs: tabOptions, ...sidebarProps } = {}, i18n = false, disableThemeSwitch = false, themeSwitch = { enabled: !disableThemeSwitch }, } = props;
    const navMode = nav.mode ?? 'auto';
    const links = getLinks(props.links ?? [], props.githubUrl);
    const tabs = useMemo(() => {
        if (Array.isArray(tabOptions)) {
            return tabOptions;
        }
        if (typeof tabOptions === 'object') {
            return getSidebarTabs(props.tree, tabOptions);
        }
        if (tabOptions !== false) {
            return getSidebarTabs(props.tree);
        }
        return [];
    }, [tabOptions, props.tree]);
    function sidebar() {
        const { banner, footer, components, collapsible = true, prefetch, defaultOpenLevel, ...rest } = sidebarProps;
        const iconLinks = links.filter((item) => item.type === 'icon');
        const rootToggle = (_jsxs(_Fragment, { children: [tabMode === 'sidebar' && tabs.length > 0 && (_jsx(RootToggle, { className: "mb-2", options: tabs })), tabMode === 'navbar' && tabs.length > 0 && (_jsx(RootToggle, { options: tabs, className: "lg:hidden" }))] }));
        const sidebarNav = (_jsxs("div", { className: "flex justify-between", children: [_jsx(Link, { href: nav.url ?? '/', className: "inline-flex items-center gap-2.5 font-medium", children: nav.title }), collapsible && (_jsx(SidebarCollapseTrigger, { className: cn(buttonVariants({
                        color: 'ghost',
                        size: 'icon-sm',
                        className: 'mt-px mb-auto text-fd-muted-foreground',
                    })), children: _jsx(SidebarIcon, {}) }))] }));
        const viewport = (_jsxs(SidebarViewport, { children: [links
                    .filter((item) => item.type !== 'icon')
                    .map((item, i, arr) => (_jsx(SidebarLinkItem, { item: item, className: cn('lg:hidden', i === arr.length - 1 && 'mb-4') }, i))), _jsx(SidebarPageTree, { components: components })] }));
        const content = (_jsxs(SidebarContent, { ...rest, className: cn(navMode === 'top'
                ? 'border-e-0 bg-transparent'
                : '[--fd-nav-height:0px]', rest.className), children: [_jsxs(HideIfEmpty, { as: SidebarHeader, children: [navMode === 'auto' && sidebarNav, nav.children, banner, rootToggle] }), viewport, _jsxs(HideIfEmpty, { as: SidebarFooter, className: "flex flex-row text-fd-muted-foreground items-center", children: [iconLinks.map((item, i) => (_jsx(BaseLinkItem, { item: item, className: cn(buttonVariants({
                                size: 'icon-sm',
                                color: 'ghost',
                                className: 'lg:hidden',
                            })), "aria-label": item.label, children: item.icon }, i))), footer] })] }));
        const mobile = (_jsxs(SidebarContentMobile, { ...rest, children: [_jsxs(SidebarHeader, { children: [_jsx(SidebarTrigger, { className: cn(buttonVariants({
                                size: 'icon-sm',
                                color: 'ghost',
                                className: 'ms-auto text-fd-muted-foreground',
                            })), children: _jsx(X, {}) }), banner, rootToggle] }), viewport, _jsxs(HideIfEmpty, { as: SidebarFooter, className: "flex flex-row items-center justify-end", children: [iconLinks.map((item, i) => (_jsx(BaseLinkItem, { item: item, className: cn(buttonVariants({
                                size: 'icon-sm',
                                color: 'ghost',
                            }), 'text-fd-muted-foreground lg:hidden', i === iconLinks.length - 1 && 'me-auto'), "aria-label": item.label, children: item.icon }, i))), i18n ? (_jsx(LanguageToggle, { children: _jsx(Languages, { className: "size-4.5 text-fd-muted-foreground" }) })) : null, themeSwitch.enabled !== false &&
                            (themeSwitch.component ?? (_jsx(ThemeToggle, { mode: themeSwitch.mode ?? 'light-dark-system' }))), footer] })] }));
        return (_jsx(Sidebar, { defaultOpenLevel: defaultOpenLevel, prefetch: prefetch, Content: content, Mobile: mobile }));
    }
    return (_jsx(TreeContextProvider, { tree: props.tree, children: _jsx(NavProvider, { transparentMode: transparentMode, children: _jsxs(LayoutBody, { ...props.containerProps, className: cn('md:[--fd-sidebar-width:286px] xl:[--fd-toc-width:286px]', props.containerProps?.className), children: [sidebar(), _jsx(DocsNavbar, { ...props, links: links, tabs: tabMode == 'navbar' ? tabs : [] }), props.children] }) }) }));
}
function DocsNavbar({ links, tabs, searchToggle = {}, themeSwitch = {}, nav = {}, ...props }) {
    const navMode = nav.mode ?? 'auto';
    const sidebarCollapsible = props.sidebar?.collapsible ?? true;
    return (_jsxs(Navbar, { mode: navMode, className: cn('on-root:[--fd-nav-height:56px] md:on-root:[--fd-nav-height:64px]', tabs.length > 0 && 'lg:on-root:[--fd-nav-height:104px]'), children: [_jsxs("div", { className: cn('flex border-b px-4 gap-2 flex-1 md:px-6', navMode === 'top' && 'ps-7'), children: [_jsxs("div", { className: cn('items-center', navMode === 'top' && 'flex flex-1', navMode === 'auto' && [
                            'hidden max-md:flex',
                            sidebarCollapsible && 'has-data-[collapsed=true]:md:flex',
                        ]), children: [sidebarCollapsible && navMode === 'auto' && (_jsx(SidebarCollapseTrigger, { className: cn(buttonVariants({
                                    color: 'ghost',
                                    size: 'icon-sm',
                                }), 'text-fd-muted-foreground data-[collapsed=false]:hidden max-md:hidden'), children: _jsx(SidebarIcon, {}) })), _jsx(Link, { href: nav.url ?? '/', className: cn('inline-flex items-center gap-2.5 font-semibold', navMode === 'auto' && 'md:hidden'), children: nav.title })] }), searchToggle.enabled !== false &&
                        (searchToggle.components?.lg ? (_jsx("div", { className: cn('w-full my-auto max-md:hidden', navMode === 'top' ? 'rounded-xl max-w-sm' : 'max-w-[240px]'), children: searchToggle.components.lg })) : (_jsx(LargeSearchToggle, { hideIfDisabled: true, className: cn('w-full my-auto max-md:hidden', navMode === 'top'
                                ? 'rounded-xl max-w-sm ps-2.5'
                                : 'max-w-[240px]') }))), _jsxs("div", { className: "flex flex-1 items-center justify-end md:gap-2", children: [_jsx("div", { className: "flex items-center gap-6 empty:hidden max-lg:hidden", children: links
                                    .filter((item) => item.type !== 'icon')
                                    .map((item, i) => (_jsx(NavbarLinkItem, { item: item, className: "text-sm text-fd-muted-foreground transition-colors hover:text-fd-accent-foreground data-[active=true]:text-fd-primary" }, i))) }), nav.children, links
                                .filter((item) => item.type === 'icon')
                                .map((item, i) => (_jsx(BaseLinkItem, { item: item, className: cn(buttonVariants({ size: 'icon-sm', color: 'ghost' }), 'text-fd-muted-foreground max-lg:hidden'), "aria-label": item.label, children: item.icon }, i))), _jsxs("div", { className: "flex items-center md:hidden", children: [searchToggle.enabled !== false &&
                                        (searchToggle.components?.sm ?? (_jsx(SearchToggle, { hideIfDisabled: true, className: "p-2" }))), _jsx(NavbarSidebarTrigger, { className: "p-2 -me-1.5" })] }), _jsxs("div", { className: "flex items-center gap-2 max-md:hidden", children: [props.i18n ? (_jsx(LanguageToggle, { children: _jsx(Languages, { className: "size-4.5 text-fd-muted-foreground" }) })) : null, themeSwitch.enabled !== false &&
                                        (themeSwitch.component ?? (_jsx(ThemeToggle, { mode: themeSwitch.mode ?? 'light-dark-system' }))), sidebarCollapsible && navMode === 'top' && (_jsx(SidebarCollapseTrigger, { className: cn(buttonVariants({
                                            color: 'secondary',
                                            size: 'icon-sm',
                                        }), 'text-fd-muted-foreground rounded-full -me-1.5'), children: _jsx(SidebarIcon, {}) }))] })] })] }), tabs.length > 0 && (_jsx(LayoutTabs, { className: cn('border-b px-6 h-10 max-lg:hidden', navMode === 'top' && 'ps-7'), options: tabs }))] }));
}
function NavbarLinkItem({ item, ...props }) {
    if (item.type === 'menu') {
        return (_jsxs(Popover, { children: [_jsxs(PopoverTrigger, { ...props, className: cn('inline-flex items-center gap-1.5 has-data-[active=true]:text-fd-primary', props.className), children: [item.url ? (_jsx(BaseLinkItem, { item: item, children: item.text })) : (item.text), _jsx(ChevronDown, { className: "size-3" })] }), _jsx(PopoverContent, { className: "flex flex-col", children: item.items.map((child, i) => {
                        if (child.type === 'custom')
                            return _jsx(Fragment, { children: child.children }, i);
                        return (_jsxs(BaseLinkItem, { item: child, className: "inline-flex items-center gap-2 rounded-md p-2 text-start hover:bg-fd-accent hover:text-fd-accent-foreground data-[active=true]:text-fd-primary [&_svg]:size-4", children: [child.icon, child.text] }, i));
                    }) })] }));
    }
    if (item.type === 'custom')
        return item.children;
    return (_jsx(BaseLinkItem, { item: item, ...props, children: item.text }));
}
function SidebarLinkItem({ item, ...props }) {
    if (item.type === 'menu')
        return (_jsxs(SidebarFolder, { ...props, children: [item.url ? (_jsxs(SidebarFolderLink, { href: item.url, external: item.external, children: [item.icon, item.text] })) : (_jsxs(SidebarFolderTrigger, { children: [item.icon, item.text] })), _jsx(SidebarFolderContent, { children: item.items.map((child, i) => (_jsx(SidebarLinkItem, { item: child }, i))) })] }));
    if (item.type === 'custom')
        return _jsx("div", { ...props, children: item.children });
    return (_jsx(SidebarItem, { href: item.url, icon: item.icon, external: item.external, ...props, children: item.text }));
}
export { Navbar, NavbarSidebarTrigger };
