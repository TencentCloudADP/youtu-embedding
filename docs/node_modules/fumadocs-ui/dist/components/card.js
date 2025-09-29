import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import Link from 'fumadocs-core/link';
import { cn } from '../utils/cn.js';
export function Cards(props) {
    return (_jsx("div", { ...props, className: cn('grid grid-cols-2 gap-3 @container', props.className), children: props.children }));
}
export function Card({ icon, title, description, ...props }) {
    const E = props.href ? Link : 'div';
    return (_jsxs(E, { ...props, "data-card": true, className: cn('block rounded-xl border bg-fd-card p-4 text-fd-card-foreground transition-colors @max-lg:col-span-full', props.href && 'hover:bg-fd-accent/80', props.className), children: [icon ? (_jsx("div", { className: "not-prose mb-2 w-fit shadow-md rounded-lg border bg-fd-muted p-1.5 text-fd-muted-foreground [&_svg]:size-4", children: icon })) : null, _jsx("h3", { className: "not-prose mb-1 text-sm font-medium", children: title }), description ? (_jsx("p", { className: "!my-0 text-sm text-fd-muted-foreground", children: description })) : null, _jsx("div", { className: "text-sm text-fd-muted-foreground prose-no-margin empty:hidden", children: props.children })] }));
}
