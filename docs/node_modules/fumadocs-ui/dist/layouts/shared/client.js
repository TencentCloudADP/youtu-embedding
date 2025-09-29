'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { usePathname } from 'fumadocs-core/framework';
import { isActive } from '../../utils/is-active.js';
import Link from 'fumadocs-core/link';
export function BaseLinkItem({ ref, item, ...props }) {
    const pathname = usePathname();
    const activeType = item.active ?? 'url';
    const active = activeType !== 'none' &&
        isActive(item.url, pathname, activeType === 'nested-url');
    return (_jsx(Link, { ref: ref, href: item.url, external: item.external, ...props, "data-active": active, children: props.children }));
}
