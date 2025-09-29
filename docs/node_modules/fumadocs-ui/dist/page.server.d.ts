import type { HTMLAttributes } from 'react';
import type { LoaderConfig, LoaderOutput, Page } from 'fumadocs-core/source';
import { type PageTree } from 'fumadocs-core/server';
/**
 * @deprecated use https://fumadocs.vercel.app/docs/ui/markdown#further-reading-section instead
 */
export declare function DocsCategory({ page, from, tree: forcedTree, ...props }: HTMLAttributes<HTMLDivElement> & {
    page: Page;
    from: LoaderOutput<LoaderConfig>;
    tree?: PageTree.Root;
}): import("react/jsx-runtime").JSX.Element | null;
export * from './page.js';
//# sourceMappingURL=page.server.d.ts.map