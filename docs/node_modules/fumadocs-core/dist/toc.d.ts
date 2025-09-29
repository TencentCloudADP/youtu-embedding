import * as react from 'react';
import { ReactNode, RefObject, AnchorHTMLAttributes } from 'react';
import { T as TableOfContents } from './get-toc-Cr2URuiP.js';
import 'unified';
import 'vfile';

/**
 * The estimated active heading ID
 */
declare function useActiveAnchor(): string | undefined;
/**
 * The id of visible anchors
 */
declare function useActiveAnchors(): string[];
interface AnchorProviderProps {
    toc: TableOfContents;
    /**
     * Only accept one active item at most
     *
     * @defaultValue true
     */
    single?: boolean;
    children?: ReactNode;
}
interface ScrollProviderProps {
    /**
     * Scroll into the view of container when active
     */
    containerRef: RefObject<HTMLElement | null>;
    children?: ReactNode;
}
declare function ScrollProvider({ containerRef, children, }: ScrollProviderProps): React.ReactElement;
declare function AnchorProvider({ toc, single, children, }: AnchorProviderProps): React.ReactElement;
interface TOCItemProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> {
    href: string;
    onActiveChange?: (v: boolean) => void;
}
declare const TOCItem: react.ForwardRefExoticComponent<TOCItemProps & react.RefAttributes<HTMLAnchorElement>>;

export { AnchorProvider, type AnchorProviderProps, ScrollProvider, type ScrollProviderProps, TOCItem, type TOCItemProps, useActiveAnchor, useActiveAnchors };
