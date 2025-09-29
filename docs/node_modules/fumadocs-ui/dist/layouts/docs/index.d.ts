import type { PageTree } from 'fumadocs-core/server';
import { type ComponentProps, type HTMLAttributes, type ReactNode } from 'react';
import { type SidebarComponents, type SidebarProps, SidebarTrigger } from '../../components/layout/sidebar.js';
import { type Option } from '../../components/layout/root-toggle.js';
import { type BaseLayoutProps, type LinkItemType } from '../../layouts/shared/index.js';
import { CollapsibleControl, Navbar } from '../../layouts/docs/client.js';
import { type GetSidebarTabsOptions } from '../../utils/get-sidebar-tabs.js';
export interface DocsLayoutProps extends BaseLayoutProps {
    tree: PageTree.Root;
    sidebar?: SidebarOptions;
    /**
     * Props for the `div` container
     */
    containerProps?: HTMLAttributes<HTMLDivElement>;
}
interface SidebarOptions extends ComponentProps<'aside'>, Pick<SidebarProps, 'defaultOpenLevel' | 'prefetch'> {
    enabled?: boolean;
    component?: ReactNode;
    components?: Partial<SidebarComponents>;
    /**
     * Root Toggle options
     */
    tabs?: Option[] | GetSidebarTabsOptions | false;
    banner?: ReactNode;
    footer?: ReactNode;
    /**
     * Support collapsing the sidebar on desktop mode
     *
     * @defaultValue true
     */
    collapsible?: boolean;
}
export declare function DocsLayout({ nav: { transparentMode, ...nav }, sidebar: { tabs: sidebarTabs, enabled: sidebarEnabled, ...sidebarProps }, searchToggle, disableThemeSwitch, themeSwitch, i18n, children, ...props }: DocsLayoutProps): import("react/jsx-runtime").JSX.Element;
export { CollapsibleControl, Navbar, SidebarTrigger, type LinkItemType };
//# sourceMappingURL=index.d.ts.map