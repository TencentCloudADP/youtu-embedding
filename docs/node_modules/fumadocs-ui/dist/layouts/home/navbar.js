'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { cva } from 'class-variance-authority';
import Link from 'fumadocs-core/link';
import { cn } from '../../utils/cn.js';
import { BaseLinkItem } from '../../layouts/shared/index.js';
import { NavigationMenu, NavigationMenuContent, NavigationMenuItem, NavigationMenuLink, NavigationMenuList, NavigationMenuTrigger, NavigationMenuViewport, } from '../../components/ui/navigation-menu.js';
import { useNav } from '../../contexts/layout.js';
import { buttonVariants } from '../../components/ui/button.js';
const navItemVariants = cva('inline-flex items-center gap-1 p-2 text-fd-muted-foreground transition-colors hover:text-fd-accent-foreground data-[active=true]:text-fd-primary [&_svg]:size-4');
export function Navbar(props) {
    const [value, setValue] = useState('');
    const { isTransparent } = useNav();
    return (_jsx(NavigationMenu, { value: value, onValueChange: setValue, asChild: true, children: _jsxs("header", { id: "nd-nav", ...props, className: cn('fixed top-(--fd-banner-height) z-40 left-0 right-(--removed-body-scroll-bar-size,0) backdrop-blur-lg border-b transition-colors *:mx-auto *:max-w-fd-container', value.length > 0 && 'max-lg:shadow-lg max-lg:rounded-b-2xl', (!isTransparent || value.length > 0) && 'bg-fd-background/80', props.className), children: [_jsx(NavigationMenuList, { className: "flex h-14 w-full items-center px-4", asChild: true, children: _jsx("nav", { children: props.children }) }), _jsx(NavigationMenuViewport, {})] }) }));
}
export const NavbarMenu = NavigationMenuItem;
export function NavbarMenuContent(props) {
    return (_jsx(NavigationMenuContent, { ...props, className: cn('grid grid-cols-1 gap-2 p-4 md:grid-cols-2 lg:grid-cols-3', props.className), children: props.children }));
}
export function NavbarMenuTrigger(props) {
    return (_jsx(NavigationMenuTrigger, { ...props, className: cn(navItemVariants(), 'rounded-md', props.className), children: props.children }));
}
export function NavbarMenuLink(props) {
    return (_jsx(NavigationMenuLink, { asChild: true, children: _jsx(Link, { ...props, className: cn('flex flex-col gap-2 rounded-lg border bg-fd-card p-3 transition-colors hover:bg-fd-accent/80 hover:text-fd-accent-foreground', props.className), children: props.children }) }));
}
const linkVariants = cva('', {
    variants: {
        variant: {
            main: navItemVariants(),
            button: buttonVariants({
                color: 'secondary',
                className: 'gap-1.5 [&_svg]:size-4',
            }),
            icon: buttonVariants({
                color: 'ghost',
                size: 'icon',
            }),
        },
    },
    defaultVariants: {
        variant: 'main',
    },
});
export function NavbarLink({ item, variant, ...props }) {
    return (_jsx(NavigationMenuItem, { children: _jsx(NavigationMenuLink, { asChild: true, children: _jsx(BaseLinkItem, { ...props, item: item, className: cn(linkVariants({ variant }), props.className), children: props.children }) }) }));
}
