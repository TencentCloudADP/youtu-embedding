import { type ComponentProps, type ReactNode } from 'react';
import { type BreadcrumbProps, type FooterProps } from './layouts/docs/page.js';
import type { AnchorProviderProps } from 'fumadocs-core/toc';
import type { TOCItemType } from 'fumadocs-core/server';
interface EditOnGitHubOptions extends Omit<ComponentProps<'a'>, 'href' | 'children'> {
    owner: string;
    repo: string;
    /**
     * SHA or ref (branch or tag) name.
     *
     * @defaultValue main
     */
    sha?: string;
    /**
     * File path in the repo
     */
    path: string;
}
interface BreadcrumbOptions extends BreadcrumbProps {
    enabled: boolean;
    component: ReactNode;
    /**
     * Show the full path to the current page
     *
     * @defaultValue false
     * @deprecated use `includePage` instead
     */
    full?: boolean;
}
interface FooterOptions extends FooterProps {
    enabled: boolean;
    component: ReactNode;
}
export interface DocsPageProps {
    toc?: TOCItemType[];
    tableOfContent?: Partial<TableOfContentOptions>;
    tableOfContentPopover?: Partial<TableOfContentPopoverOptions>;
    /**
     * Extend the page to fill all available space
     *
     * @defaultValue false
     */
    full?: boolean;
    /**
     * Replace or disable breadcrumb
     */
    breadcrumb?: Partial<BreadcrumbOptions>;
    /**
     * Footer navigation, you can disable it by passing `false`
     */
    footer?: Partial<FooterOptions>;
    editOnGithub?: EditOnGitHubOptions;
    lastUpdate?: Date | string | number;
    container?: ComponentProps<'div'>;
    article?: ComponentProps<'article'>;
    children?: ReactNode;
}
type TableOfContentOptions = Pick<AnchorProviderProps, 'single'> & {
    /**
     * Custom content in TOC container, before the main TOC
     */
    header?: ReactNode;
    /**
     * Custom content in TOC container, after the main TOC
     */
    footer?: ReactNode;
    enabled: boolean;
    component: ReactNode;
    /**
     * @defaultValue 'normal'
     */
    style?: 'normal' | 'clerk';
};
type TableOfContentPopoverOptions = Omit<TableOfContentOptions, 'single'>;
export declare function DocsPage({ editOnGithub, breadcrumb: { enabled: breadcrumbEnabled, component: breadcrumb, ...breadcrumbProps }, footer, lastUpdate, container, full, tableOfContentPopover: { enabled: tocPopoverEnabled, component: tocPopover, ...tocPopoverOptions }, tableOfContent: { enabled: tocEnabled, component: tocReplace, ...tocOptions }, toc, article, children, }: DocsPageProps): import("react/jsx-runtime").JSX.Element;
export declare function EditOnGitHub(props: ComponentProps<'a'>): import("react/jsx-runtime").JSX.Element;
/**
 * Add typography styles
 */
export declare const DocsBody: import("react").ForwardRefExoticComponent<Omit<import("react").DetailedHTMLProps<import("react").HTMLAttributes<HTMLDivElement>, HTMLDivElement>, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
export declare const DocsDescription: import("react").ForwardRefExoticComponent<Omit<import("react").DetailedHTMLProps<import("react").HTMLAttributes<HTMLParagraphElement>, HTMLParagraphElement>, "ref"> & import("react").RefAttributes<HTMLParagraphElement>>;
export declare const DocsTitle: import("react").ForwardRefExoticComponent<Omit<import("react").DetailedHTMLProps<import("react").HTMLAttributes<HTMLHeadingElement>, HTMLHeadingElement>, "ref"> & import("react").RefAttributes<HTMLHeadingElement>>;
/**
 * For separate MDX page
 */
export declare function withArticle(props: ComponentProps<'main'>): ReactNode;
export {};
//# sourceMappingURL=page.d.ts.map