import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { CircleCheck, CircleX, Info, TriangleAlert } from '../icons.js';
import { forwardRef } from 'react';
import { cn } from '../utils/cn.js';
const iconClass = 'size-5 -me-0.5 fill-(--callout-color) text-fd-card';
export const Callout = forwardRef(({ className, children, title, type = 'info', icon, ...props }, ref) => {
    if (type === 'warn')
        type = 'warning';
    if (type === 'tip')
        type = 'info';
    return (_jsxs("div", { ref: ref, className: cn('flex gap-2 my-4 rounded-xl border bg-fd-card p-3 ps-1 text-sm text-fd-card-foreground shadow-md', className), ...props, style: {
            '--callout-color': `var(--color-fd-${type}, var(--color-fd-muted))`,
            ...props.style,
        }, children: [_jsx("div", { role: "none", className: "w-0.5 bg-(--callout-color)/50 rounded-sm" }), icon ??
                {
                    info: _jsx(Info, { className: iconClass }),
                    warning: _jsx(TriangleAlert, { className: iconClass }),
                    error: _jsx(CircleX, { className: iconClass }),
                    success: _jsx(CircleCheck, { className: iconClass }),
                }[type], _jsxs("div", { className: "flex flex-col gap-2 min-w-0 flex-1", children: [title && _jsx("p", { className: "font-medium !my-0", children: title }), _jsx("div", { className: "text-fd-muted-foreground prose-no-margin empty:hidden", children: children })] })] }));
});
Callout.displayName = 'Callout';
