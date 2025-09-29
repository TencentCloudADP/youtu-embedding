import type { TOCItemType } from 'fumadocs-core/server';
import { Collapsible } from './ui/collapsible.js';
import { ComponentProps } from 'react';
export interface InlineTocProps extends ComponentProps<typeof Collapsible> {
    items: TOCItemType[];
}
export declare function InlineTOC({ items, ...props }: InlineTocProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=inline-toc.d.ts.map