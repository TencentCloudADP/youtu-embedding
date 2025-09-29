import { RootProvider as BaseProvider } from './base.js';
import type { ComponentProps } from 'react';
export declare function RootProvider(props: ComponentProps<typeof BaseProvider>): import("react/jsx-runtime").JSX.Element;
export { useI18n, I18nLabel } from '../contexts/i18n.js';
export { SearchProvider, SearchOnly, useSearchContext, type SearchProviderProps, } from '../contexts/search.js';
export { SidebarProvider, useSidebar } from '../contexts/sidebar.js';
export { useTreePath, useTreeContext, TreeContextProvider, } from '../contexts/tree.js';
export { useNav, NavProvider, type NavProviderProps, type PageStyles, StylesProvider, usePageStyles, } from '../contexts/layout.js';
//# sourceMappingURL=index.d.ts.map