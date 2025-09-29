'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { useMemo, useRef, useState, } from 'react';
import { createContext, usePathname } from 'fumadocs-core/framework';
import { useOnChange } from 'fumadocs-core/utils/use-on-change';
const SidebarContext = createContext('SidebarContext');
export function useSidebar() {
    return SidebarContext.use();
}
export function SidebarProvider({ children, }) {
    const closeOnRedirect = useRef(true);
    const [open, setOpen] = useState(false);
    const [collapsed, setCollapsed] = useState(false);
    const pathname = usePathname();
    useOnChange(pathname, () => {
        if (closeOnRedirect.current) {
            setOpen(false);
        }
        closeOnRedirect.current = true;
    });
    return (_jsx(SidebarContext.Provider, { value: useMemo(() => ({
            open,
            setOpen,
            collapsed,
            setCollapsed,
            closeOnRedirect,
        }), [open, collapsed]), children: children }));
}
