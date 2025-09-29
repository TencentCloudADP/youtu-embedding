import type { AnchorHTMLAttributes, FC, HTMLAttributes, ImgHTMLAttributes, TableHTMLAttributes } from 'react';
import { Card, Cards } from './components/card.js';
import { CodeBlockTab, CodeBlockTabs, CodeBlockTabsList, CodeBlockTabsTrigger } from './components/codeblock.js';
declare function Image(props: ImgHTMLAttributes<HTMLImageElement> & {
    sizes?: string;
}): import("react/jsx-runtime").JSX.Element;
declare function Table(props: TableHTMLAttributes<HTMLTableElement>): import("react/jsx-runtime").JSX.Element;
declare const defaultMdxComponents: {
    CodeBlockTab: typeof CodeBlockTab;
    CodeBlockTabs: typeof CodeBlockTabs;
    CodeBlockTabsList: typeof CodeBlockTabsList;
    CodeBlockTabsTrigger: typeof CodeBlockTabsTrigger;
    pre: (props: HTMLAttributes<HTMLPreElement>) => import("react/jsx-runtime").JSX.Element;
    Card: typeof Card;
    Cards: typeof Cards;
    a: FC<AnchorHTMLAttributes<HTMLAnchorElement>>;
    img: typeof Image;
    h1: (props: HTMLAttributes<HTMLHeadingElement>) => import("react/jsx-runtime").JSX.Element;
    h2: (props: HTMLAttributes<HTMLHeadingElement>) => import("react/jsx-runtime").JSX.Element;
    h3: (props: HTMLAttributes<HTMLHeadingElement>) => import("react/jsx-runtime").JSX.Element;
    h4: (props: HTMLAttributes<HTMLHeadingElement>) => import("react/jsx-runtime").JSX.Element;
    h5: (props: HTMLAttributes<HTMLHeadingElement>) => import("react/jsx-runtime").JSX.Element;
    h6: (props: HTMLAttributes<HTMLHeadingElement>) => import("react/jsx-runtime").JSX.Element;
    table: typeof Table;
    Callout: import("react").ForwardRefExoticComponent<Omit<HTMLAttributes<HTMLDivElement>, "title" | "type" | "icon"> & {
        title?: import("react").ReactNode;
        type?: "info" | "warn" | "error" | "success" | "warning";
        icon?: import("react").ReactNode;
    } & import("react").RefAttributes<HTMLDivElement>>;
};
export declare const createRelativeLink: typeof import('./mdx.server.js').createRelativeLink;
export { defaultMdxComponents as default };
//# sourceMappingURL=mdx.d.ts.map