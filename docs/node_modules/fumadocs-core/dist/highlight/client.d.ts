import { ReactNode, DependencyList } from 'react';
import { HighlightOptions } from './index.js';
import 'shiki';
import 'shiki/themes';
import 'hast-util-to-jsx-runtime';
import 'hast';

declare function useShiki(code: string, options: HighlightOptions & {
    /**
     * @deprecated no longer pre-rendered using scripts.
     */
    withPrerenderScript?: boolean;
    /**
     * Displayed before highlighter is loaded.
     *
     * @deprecated use React `Suspense` fallback instead.
     */
    loading?: ReactNode;
}, deps?: DependencyList): ReactNode;

export { useShiki };
