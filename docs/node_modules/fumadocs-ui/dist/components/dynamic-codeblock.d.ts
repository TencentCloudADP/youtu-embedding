import { type CodeBlockProps } from '../components/codeblock.js';
import type { HighlightOptionsCommon, HighlightOptionsThemes } from 'fumadocs-core/highlight';
export interface DynamicCodeblockProps {
    lang: string;
    code: string;
    /**
     * Extra props for the underlying `<CodeBlock />` component.
     *
     * Ignored if you defined your own `pre` component in `options.components`.
     */
    codeblock?: CodeBlockProps;
    /**
     * Wrap in React `<Suspense />` and provide a fallback.
     *
     * @defaultValue true
     */
    wrapInSuspense?: boolean;
    options?: Omit<HighlightOptionsCommon, 'lang'> & HighlightOptionsThemes;
}
export declare function DynamicCodeBlock({ lang, code, codeblock, options, wrapInSuspense, }: DynamicCodeblockProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=dynamic-codeblock.d.ts.map