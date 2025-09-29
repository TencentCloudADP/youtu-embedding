'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from 'react';
import { createContext } from 'fumadocs-core/framework';
/**
 * applied styles to different layout components in `Page` from layouts
 */
const StylesContext = createContext('StylesContext', {
    tocNav: 'xl:hidden',
    toc: 'max-xl:hidden',
});
export function usePageStyles() {
    return StylesContext.use();
}
export function StylesProvider({ children, ...value }) {
    return (_jsx(StylesContext.Provider, { value: value, children: children }));
}
const NavContext = createContext('NavContext', {
    isTransparent: false,
});
export function NavProvider({ transparentMode = 'none', children, }) {
    const [transparent, setTransparent] = useState(transparentMode !== 'none');
    useEffect(() => {
        if (transparentMode !== 'top')
            return;
        const listener = () => {
            setTransparent(window.scrollY < 10);
        };
        listener();
        window.addEventListener('scroll', listener);
        return () => {
            window.removeEventListener('scroll', listener);
        };
    }, [transparentMode]);
    return (_jsx(NavContext.Provider, { value: useMemo(() => ({ isTransparent: transparent }), [transparent]), children: children }));
}
export function useNav() {
    return NavContext.use();
}
