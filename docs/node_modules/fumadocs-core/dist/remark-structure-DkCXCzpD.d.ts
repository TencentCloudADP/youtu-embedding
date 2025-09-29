import { Nodes, Root } from 'mdast';
import { Transformer, PluggableList } from 'unified';
import { MdxJsxFlowElement, MdxJsxAttribute, MdxJsxExpressionAttribute } from 'mdast-util-mdx-jsx';

interface Heading {
    id: string;
    content: string;
}
interface Content {
    heading: string | undefined;
    content: string;
}
interface StructuredData {
    headings: Heading[];
    /**
     * Refer to paragraphs, a heading may contain multiple contents as well
     */
    contents: Content[];
}
interface StructureOptions {
    /**
     * Types to be scanned as content.
     *
     * @defaultValue ['heading', 'paragraph', 'blockquote', 'tableCell', 'mdxJsxFlowElement']
     */
    types?: string[] | ((node: Nodes) => boolean);
    /**
     * A list of indexable MDX attributes, either:
     *
     * - an array of attribute names.
     * - a function that determines if attribute should be indexed.
     */
    allowedMdxAttributes?: string[] | ((node: MdxJsxFlowElement, attribute: MdxJsxAttribute | MdxJsxExpressionAttribute) => boolean);
}
declare module 'mdast' {
    interface Data {
        /**
         * Get content of unserializable element
         *
         * Needed for `remarkStructure` to generate search index
         */
        _string?: string[];
    }
}
declare module 'vfile' {
    interface DataMap {
        structuredData: StructuredData;
    }
}
/**
 * Attach structured data to VFile, you can access via `vfile.data.structuredData`.
 */
declare function remarkStructure({ types, allowedMdxAttributes, }?: StructureOptions): Transformer<Root, Root>;
/**
 * Extract data from markdown/mdx content
 */
declare function structure(content: string, remarkPlugins?: PluggableList, options?: StructureOptions): StructuredData;

export { type StructuredData as S, type StructureOptions as a, remarkStructure as r, structure as s };
