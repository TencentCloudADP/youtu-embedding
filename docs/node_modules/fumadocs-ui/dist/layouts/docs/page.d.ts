import { type ComponentProps } from 'react';
import { type BreadcrumbProps, type FooterProps, PageBreadcrumb, PageFooter, PageLastUpdate, PageTOC, PageTOCPopover, PageTOCPopoverContent, PageTOCPopoverTrigger } from './page-client.js';
import type { AnchorProviderProps } from 'fumadocs-core/toc';
export declare function PageTOCTitle(props: ComponentProps<'h2'>): import("react/jsx-runtime").JSX.Element;
export declare function PageTOCItems({ variant, ...props }: ComponentProps<'div'> & {
    variant?: 'clerk' | 'normal';
}): import("react/jsx-runtime").JSX.Element;
export declare function PageTOCPopoverItems({ variant, ...props }: ComponentProps<'div'> & {
    variant?: 'clerk' | 'normal';
}): import("react/jsx-runtime").JSX.Element;
export declare function PageArticle(props: ComponentProps<'article'>): import("react/jsx-runtime").JSX.Element;
export interface RootProps extends ComponentProps<'div'> {
    toc?: Omit<AnchorProviderProps, 'children'> | false;
}
export declare function PageRoot({ toc, children, ...props }: RootProps): import("react/jsx-runtime").JSX.Element;
export { PageBreadcrumb, PageFooter, PageLastUpdate, PageTOC, PageTOCPopover, PageTOCPopoverTrigger, PageTOCPopoverContent, type FooterProps, type BreadcrumbProps, };
//# sourceMappingURL=page.d.ts.map