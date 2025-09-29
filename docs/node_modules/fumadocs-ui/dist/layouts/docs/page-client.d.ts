import { type ComponentProps } from 'react';
import type { PageTree } from 'fumadocs-core/server';
import { type BreadcrumbOptions } from 'fumadocs-core/breadcrumb';
export declare function PageTOCPopoverTrigger(props: ComponentProps<'button'>): import("react/jsx-runtime").JSX.Element;
export declare function PageTOCPopoverContent(props: ComponentProps<'div'>): import("react/jsx-runtime").JSX.Element;
export declare function PageTOCPopover(props: ComponentProps<'div'>): import("react/jsx-runtime").JSX.Element;
export declare function PageLastUpdate({ date: value, ...props }: Omit<ComponentProps<'p'>, 'children'> & {
    date: Date | string;
}): import("react/jsx-runtime").JSX.Element;
type Item = Pick<PageTree.Item, 'name' | 'description' | 'url'>;
export interface FooterProps extends ComponentProps<'div'> {
    /**
     * Items including information for the next and previous page
     */
    items?: {
        previous?: Item;
        next?: Item;
    };
}
export declare function PageFooter({ items, ...props }: FooterProps): import("react/jsx-runtime").JSX.Element;
export type BreadcrumbProps = BreadcrumbOptions & ComponentProps<'div'>;
export declare function PageBreadcrumb({ includeRoot, includeSeparator, includePage, ...props }: BreadcrumbProps): import("react/jsx-runtime").JSX.Element | null;
export declare function PageTOC(props: ComponentProps<'div'>): import("react/jsx-runtime").JSX.Element;
export {};
//# sourceMappingURL=page-client.d.ts.map