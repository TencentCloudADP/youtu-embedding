import type { TOCItemType } from 'fumadocs-core/server';
import * as Primitive from 'fumadocs-core/toc';
import { type ComponentProps } from 'react';
export declare function useTOCItems(): TOCItemType[];
export declare function TOCProvider({ toc, children, ...props }: ComponentProps<typeof Primitive.AnchorProvider>): import("react/jsx-runtime").JSX.Element;
export declare function TOCScrollArea({ ref, className, ...props }: ComponentProps<'div'>): import("react/jsx-runtime").JSX.Element;
export declare function TOCItems({ ref, className, ...props }: ComponentProps<'div'>): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=toc.d.ts.map