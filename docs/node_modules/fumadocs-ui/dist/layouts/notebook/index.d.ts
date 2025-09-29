import { type ComponentProps, type HTMLAttributes, type ReactNode } from 'react';
import { type BaseLayoutProps } from '../../layouts/shared/index.js';
import { type SidebarComponents, type SidebarProps } from '../../components/layout/sidebar.js';
import type { PageTree } from 'fumadocs-core/server';
import { Navbar, NavbarSidebarTrigger } from '../../layouts/notebook/client.js';
import { type Option } from '../../components/layout/root-toggle.js';
import { type GetSidebarTabsOptions } from '../../utils/get-sidebar-tabs.js';
export interface DocsLayoutProps extends BaseLayoutProps {
    tree: PageTree.Root;
    tabMode?: 'sidebar' | 'navbar';
    nav?: BaseLayoutProps['nav'] & {
        mode?: 'top' | 'auto';
    };
    sidebar?: SidebarOptions;
    containerProps?: HTMLAttributes<HTMLDivElement>;
}
interface SidebarOptions extends ComponentProps<'aside'>, Pick<SidebarProps, 'defaultOpenLevel' | 'prefetch'> {
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
export declare function DocsLayout(props: DocsLayoutProps): import("react/jsx-runtime").JSX.Element;
export { Navbar, NavbarSidebarTrigger };
//# sourceMappingURL=index.d.ts.map