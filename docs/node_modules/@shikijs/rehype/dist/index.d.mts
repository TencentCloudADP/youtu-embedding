import { LanguageInput } from '@shikijs/types';
import { Root } from 'hast';
import { BuiltinLanguage } from 'shiki';
import { Plugin } from 'unified';
import { R as RehypeShikiCoreOptions } from './shared/rehype.DcMmi29I.mjs';

type RehypeShikiOptions = RehypeShikiCoreOptions & {
    /**
     * Language names to include.
     *
     * @default Object.keys(bundledLanguages)
     */
    langs?: Array<LanguageInput | BuiltinLanguage>;
    /**
     * Alias of languages
     * @example { 'my-lang': 'javascript' }
     */
    langAlias?: Record<string, string>;
};
declare const rehypeShiki: Plugin<[RehypeShikiOptions], Root>;

export { rehypeShiki as default };
export type { RehypeShikiOptions };
