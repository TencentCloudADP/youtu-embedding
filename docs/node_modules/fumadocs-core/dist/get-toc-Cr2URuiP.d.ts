import { ReactNode } from 'react';
import { PluggableList } from 'unified';
import { Compatible } from 'vfile';

interface TOCItemType {
    title: ReactNode;
    url: string;
    depth: number;
}
type TableOfContents = TOCItemType[];
/**
 * Get Table of Contents from markdown/mdx document (using remark)
 *
 * @param content - Markdown content or file
 */
declare function getTableOfContents(content: Compatible): TableOfContents;
/**
 * Get Table of Contents from markdown/mdx document (using remark)
 *
 * @param content - Markdown content or file
 * @param remarkPlugins - remark plugins to be applied first
 */
declare function getTableOfContents(content: Compatible, remarkPlugins: PluggableList): Promise<TableOfContents>;

export { type TableOfContents as T, type TOCItemType as a, getTableOfContents as g };
