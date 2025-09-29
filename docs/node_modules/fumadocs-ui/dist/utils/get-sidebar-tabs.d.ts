import type { PageTree } from 'fumadocs-core/server';
import type { ReactNode } from 'react';
export interface SidebarTab {
    /**
     * Redirect URL of the folder, usually the index page
     */
    url: string;
    icon?: ReactNode;
    title: ReactNode;
    description?: ReactNode;
    /**
     * Detect from a list of urls
     */
    urls?: Set<string>;
    unlisted?: boolean;
}
export interface GetSidebarTabsOptions {
    transform?: (option: SidebarTab, node: PageTree.Folder) => SidebarTab | null;
}
export declare function getSidebarTabs(tree: PageTree.Root, { transform }?: GetSidebarTabsOptions): SidebarTab[];
//# sourceMappingURL=get-sidebar-tabs.d.ts.map