import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link } from '../icons.js';
import { cn } from '../utils/cn.js';
export function Heading({ as, className, ...props }) {
    const As = as ?? 'h1';
    if (!props.id)
        return _jsx(As, { className: className, ...props });
    return (_jsxs(As, { className: cn('flex scroll-m-28 flex-row items-center gap-2', className), ...props, children: [_jsx("a", { "data-card": "", href: `#${props.id}`, className: "peer", children: props.children }), _jsx(Link, { "aria-label": "Link to section", className: "size-3.5 shrink-0 text-fd-muted-foreground opacity-0 transition-opacity peer-hover:opacity-100" })] }));
}
