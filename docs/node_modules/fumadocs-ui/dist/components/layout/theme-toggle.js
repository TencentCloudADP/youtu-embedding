'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { cva } from 'class-variance-authority';
import { Moon, Sun, Airplay } from '../../icons.js';
import { useTheme } from 'next-themes';
import { useLayoutEffect, useState } from 'react';
import { cn } from '../../utils/cn.js';
const itemVariants = cva('size-6.5 rounded-full p-1.5 text-fd-muted-foreground', {
    variants: {
        active: {
            true: 'bg-fd-accent text-fd-accent-foreground',
            false: 'text-fd-muted-foreground',
        },
    },
});
const full = [
    ['light', Sun],
    ['dark', Moon],
    ['system', Airplay],
];
export function ThemeToggle({ className, mode = 'light-dark', ...props }) {
    const { setTheme, theme, resolvedTheme } = useTheme();
    const [mounted, setMounted] = useState(false);
    useLayoutEffect(() => {
        setMounted(true);
    }, []);
    const container = cn('inline-flex items-center rounded-full border p-1', className);
    if (mode === 'light-dark') {
        const value = mounted ? resolvedTheme : null;
        return (_jsx("button", { className: container, "aria-label": `Toggle Theme`, onClick: () => setTheme(value === 'light' ? 'dark' : 'light'), "data-theme-toggle": "", ...props, children: full.map(([key, Icon]) => {
                if (key === 'system')
                    return;
                return (_jsx(Icon, { fill: "currentColor", className: cn(itemVariants({ active: value === key })) }, key));
            }) }));
    }
    const value = mounted ? theme : null;
    return (_jsx("div", { className: container, "data-theme-toggle": "", ...props, children: full.map(([key, Icon]) => (_jsx("button", { "aria-label": key, className: cn(itemVariants({ active: value === key })), onClick: () => setTheme(key), children: _jsx(Icon, { className: "size-full", fill: "currentColor" }) }, key))) }));
}
