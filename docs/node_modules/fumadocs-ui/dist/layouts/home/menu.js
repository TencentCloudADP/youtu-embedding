'use client';
import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { BaseLinkItem } from '../../layouts/shared/index.js';
import { cn } from '../../utils/cn.js';
import { NavigationMenuContent, NavigationMenuItem, NavigationMenuLink, NavigationMenuTrigger, } from '../../components/ui/navigation-menu.js';
import Link from 'fumadocs-core/link';
import { cva } from 'class-variance-authority';
import { buttonVariants } from '../../components/ui/button.js';
const menuItemVariants = cva('', {
    variants: {
        variant: {
            main: 'inline-flex items-center gap-2 py-1.5 transition-colors hover:text-fd-popover-foreground/50 data-[active=true]:font-medium data-[active=true]:text-fd-primary [&_svg]:size-4',
            icon: buttonVariants({
                size: 'icon',
                color: 'ghost',
            }),
            button: buttonVariants({
                color: 'secondary',
                className: 'gap-1.5 [&_svg]:size-4',
            }),
        },
    },
    defaultVariants: {
        variant: 'main',
    },
});
export function MenuLinkItem({ item, ...props }) {
    if (item.type === 'custom')
        return _jsx("div", { className: cn('grid', props.className), children: item.children });
    if (item.type === 'menu') {
        const header = (_jsxs(_Fragment, { children: [item.icon, item.text] }));
        return (_jsxs("div", { className: cn('mb-4 flex flex-col', props.className), children: [_jsx("p", { className: "mb-1 text-sm text-fd-muted-foreground", children: item.url ? (_jsx(NavigationMenuLink, { asChild: true, children: _jsx(Link, { href: item.url, external: item.external, children: header }) })) : (header) }), item.items.map((child, i) => (_jsx(MenuLinkItem, { item: child }, i)))] }));
    }
    return (_jsx(NavigationMenuLink, { asChild: true, children: _jsxs(BaseLinkItem, { item: item, className: cn(menuItemVariants({ variant: item.type }), props.className), "aria-label": item.type === 'icon' ? item.label : undefined, children: [item.icon, item.type === 'icon' ? undefined : item.text] }) }));
}
export const Menu = NavigationMenuItem;
export function MenuTrigger({ enableHover = false, ...props }) {
    return (_jsx(NavigationMenuTrigger, { ...props, onPointerMove: enableHover ? undefined : (e) => e.preventDefault(), children: props.children }));
}
export function MenuContent(props) {
    return (_jsx(NavigationMenuContent, { ...props, className: cn('flex flex-col p-4', props.className), children: props.children }));
}
