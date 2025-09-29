import { type ComponentProps, type FC, type ReactNode } from 'react';
import { type LinkProps } from 'fumadocs-core/link';
import { type ScrollAreaProps } from '@radix-ui/react-scroll-area';
import type { CollapsibleContentProps, CollapsibleTriggerProps } from '@radix-ui/react-collapsible';
import type { PageTree } from 'fumadocs-core/server';
export interface SidebarProps {
    /**
     * Open folders by default if their level is lower or equal to a specific level
     * (Starting from 1)
     *
     * @defaultValue 0
     */
    defaultOpenLevel?: number;
    /**
     * Prefetch links
     *
     * @defaultValue true
     */
    prefetch?: boolean;
    /**
     * Children to render
     */
    Content: ReactNode;
    /**
     * Alternative children for mobile
     */
    Mobile?: ReactNode;
}
export declare function Sidebar({ defaultOpenLevel, prefetch, Mobile, Content, }: SidebarProps): import("react/jsx-runtime").JSX.Element;
export declare function SidebarContent(props: ComponentProps<'aside'>): import("react/jsx-runtime").JSX.Element;
export declare function SidebarContentMobile({ className, children, ...props }: ComponentProps<'aside'>): import("react/jsx-runtime").JSX.Element;
export declare function SidebarHeader(props: ComponentProps<'div'>): import("react/jsx-runtime").JSX.Element;
export declare function SidebarFooter(props: ComponentProps<'div'>): import("react/jsx-runtime").JSX.Element;
export declare function SidebarViewport(props: ScrollAreaProps): import("react/jsx-runtime").JSX.Element;
export declare function SidebarSeparator(props: ComponentProps<'p'>): import("react/jsx-runtime").JSX.Element;
export declare function SidebarItem({ icon, ...props }: LinkProps & {
    icon?: ReactNode;
}): import("react/jsx-runtime").JSX.Element;
export declare function SidebarFolder({ defaultOpen, ...props }: ComponentProps<'div'> & {
    defaultOpen?: boolean;
}): import("react/jsx-runtime").JSX.Element;
export declare function SidebarFolderTrigger({ className, ...props }: CollapsibleTriggerProps): import("react/jsx-runtime").JSX.Element;
export declare function SidebarFolderLink(props: LinkProps): import("react/jsx-runtime").JSX.Element;
export declare function SidebarFolderContent(props: CollapsibleContentProps): import("react/jsx-runtime").JSX.Element;
export declare function SidebarTrigger({ children, ...props }: ComponentProps<'button'>): import("react/jsx-runtime").JSX.Element;
export declare function SidebarCollapseTrigger(props: ComponentProps<'button'>): import("react/jsx-runtime").JSX.Element;
export interface SidebarComponents {
    Item: FC<{
        item: PageTree.Item;
    }>;
    Folder: FC<{
        item: PageTree.Folder;
        level: number;
        children: ReactNode;
    }>;
    Separator: FC<{
        item: PageTree.Separator;
    }>;
}
/**
 * Render sidebar items from page tree
 */
export declare function SidebarPageTree(props: {
    components?: Partial<SidebarComponents>;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=sidebar.d.ts.map