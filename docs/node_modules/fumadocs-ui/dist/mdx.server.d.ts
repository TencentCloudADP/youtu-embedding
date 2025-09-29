import type { LoaderConfig, LoaderOutput, Page } from 'fumadocs-core/source';
import type { ComponentProps, FC } from 'react';
import defaultMdxComponents from './mdx.js';
/**
 * Extend the default Link component to resolve relative file paths in `href`.
 *
 * @param page the current page
 * @param source the source object
 * @param OverrideLink The component to override from
 */
export declare function createRelativeLink(source: LoaderOutput<LoaderConfig>, page: Page, OverrideLink?: FC<ComponentProps<'a'>>): FC<ComponentProps<'a'>>;
export { defaultMdxComponents as default };
//# sourceMappingURL=mdx.server.d.ts.map