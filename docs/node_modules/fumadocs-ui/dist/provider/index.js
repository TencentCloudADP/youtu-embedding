'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { RootProvider as BaseProvider } from './base.js';
import { NextProvider } from 'fumadocs-core/framework/next';
export function RootProvider(props) {
    return (_jsx(NextProvider, { children: _jsx(BaseProvider, { ...props, children: props.children }) }));
}
export { useI18n, I18nLabel } from '../contexts/i18n.js';
export { SearchProvider, SearchOnly, useSearchContext, } from '../contexts/search.js';
export { SidebarProvider, useSidebar } from '../contexts/sidebar.js';
export { useTreePath, useTreeContext, TreeContextProvider, } from '../contexts/tree.js';
export { useNav, NavProvider, StylesProvider, usePageStyles, } from '../contexts/layout.js';
