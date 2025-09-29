import { Root, Heading, Code } from 'mdast';
import { Transformer, Processor } from 'unified';
import { MdxJsxFlowElement } from 'mdast-util-mdx-jsx';

declare module 'mdast' {
    interface HeadingData extends Data {
        hProperties?: {
            id?: string;
        };
    }
}
interface RemarkHeadingOptions {
    slug?: (root: Root, heading: Heading, text: string) => string;
    /**
     * Allow custom headings ids
     *
     * @defaultValue true
     */
    customId?: boolean;
    /**
     * Attach an array of `TOCItemType` to `file.data.toc`
     *
     * @defaultValue true
     */
    generateToc?: boolean;
}
/**
 * Add heading ids and extract TOC
 */
declare function remarkHeading({ slug: defaultSlug, customId, generateToc, }?: RemarkHeadingOptions): Transformer<Root, Root>;

type TabType = keyof typeof Types;
interface RemarkCodeTabOptions {
    Tabs?: TabType;
    /**
     * Parse MDX in tab values
     *
     * @defaultValue false
     */
    parseMdx?: boolean;
}
declare module 'mdast' {
    interface CodeData {
        tab?: string;
    }
}
declare const Types: {
    CodeBlockTabs: {
        convert(processor: Processor, nodes: Code[], withMdx?: boolean, withParent?: boolean): MdxJsxFlowElement;
    };
    Tabs: {
        convert(processor: Processor, nodes: Code[], withMdx?: boolean, withParent?: boolean): MdxJsxFlowElement;
    };
};
declare function remarkCodeTab(this: Processor, options?: RemarkCodeTabOptions): Transformer<Root, Root>;

export { type RemarkHeadingOptions as R, type RemarkCodeTabOptions as a, remarkCodeTab as b, remarkHeading as r };
