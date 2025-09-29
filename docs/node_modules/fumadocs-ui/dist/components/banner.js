'use client';
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { X } from '../icons.js';
import { cn } from '../utils/cn.js';
import { buttonVariants } from '../components/ui/button.js';
export function Banner({ id, variant = 'normal', changeLayout = true, height = '3rem', rainbowColors = [
    'rgba(0,149,255,0.56)',
    'rgba(231,77,255,0.77)',
    'rgba(255,0,0,0.73)',
    'rgba(131,255,166,0.66)',
], ...props }) {
    const [open, setOpen] = useState(true);
    const globalKey = id ? `nd-banner-${id}` : null;
    useEffect(() => {
        if (globalKey)
            setOpen(localStorage.getItem(globalKey) !== 'true');
    }, [globalKey]);
    if (!open)
        return null;
    return (_jsxs("div", { id: id, ...props, className: cn('sticky top-0 z-40 flex flex-row items-center justify-center px-4 text-center text-sm font-medium', variant === 'normal' && 'bg-fd-secondary', variant === 'rainbow' && 'bg-fd-background', !open && 'hidden', props.className), style: {
            height,
        }, children: [changeLayout && open ? (_jsx("style", { children: globalKey
                    ? `:root:not(.${globalKey}) { --fd-banner-height: ${height}; }`
                    : `:root { --fd-banner-height: ${height}; }` })) : null, globalKey ? (_jsx("style", { children: `.${globalKey} #${id} { display: none; }` })) : null, globalKey ? (_jsx("script", { dangerouslySetInnerHTML: {
                    __html: `if (localStorage.getItem('${globalKey}') === 'true') document.documentElement.classList.add('${globalKey}');`,
                } })) : null, variant === 'rainbow'
                ? flow({
                    colors: rainbowColors,
                })
                : null, props.children, id ? (_jsx("button", { type: "button", "aria-label": "Close Banner", onClick: () => {
                    setOpen(false);
                    if (globalKey)
                        localStorage.setItem(globalKey, 'true');
                }, className: cn(buttonVariants({
                    color: 'ghost',
                    className: 'absolute end-2 top-1/2 -translate-y-1/2 text-fd-muted-foreground/50',
                    size: 'icon-sm',
                })), children: _jsx(X, {}) })) : null] }));
}
const maskImage = 'linear-gradient(to bottom,white,transparent), radial-gradient(circle at top center, white, transparent)';
function flow({ colors }) {
    return (_jsxs(_Fragment, { children: [_jsx("div", { className: "absolute inset-0 z-[-1]", style: {
                    maskImage,
                    maskComposite: 'intersect',
                    animation: 'fd-moving-banner 20s linear infinite',
                    backgroundImage: `repeating-linear-gradient(70deg, ${[...colors, colors[0]].map((color, i) => `${color} ${(i * 50) / colors.length}%`).join(', ')})`,
                    backgroundSize: '200% 100%',
                    filter: 'saturate(2)',
                } }), _jsx("style", { children: `@keyframes fd-moving-banner {
            from { background-position: 0% 0;  }
            to { background-position: 100% 0;  }
         }` })] }));
}
