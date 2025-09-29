import { CodeToHastOptionsCommon, BundledLanguage, CodeOptionsMeta, Awaitable, RegexEngine, CodeOptionsThemes, BundledHighlighterOptions, Highlighter } from 'shiki';
import { BundledTheme } from 'shiki/themes';
import { Components } from 'hast-util-to-jsx-runtime';
import { ReactNode } from 'react';
import { Root } from 'hast';

type HighlightOptionsCommon = CodeToHastOptionsCommon<BundledLanguage> & CodeOptionsMeta & {
    engine?: 'js' | 'oniguruma' | Awaitable<RegexEngine>;
    components?: Partial<Components>;
    fallbackLanguage?: BundledLanguage;
};
type HighlightOptionsThemes = CodeOptionsThemes<BundledTheme>;
type HighlightOptions = HighlightOptionsCommon & (HighlightOptionsThemes | Record<never, never>);
declare function highlightHast(code: string, options: HighlightOptions): Promise<Root>;
/**
 * Get Shiki highlighter instance of Fumadocs (mostly for internal use, don't recommend you to use it).
 *
 * @param engineType - engine type, the engine specified in `options` will only be effective when this is set to `custom`.
 * @param options - Shiki options.
 */
declare function getHighlighter(engineType: 'js' | 'oniguruma' | 'custom', options: BundledHighlighterOptions<BundledLanguage, BundledTheme>): Promise<Highlighter>;
declare function highlight(code: string, options: HighlightOptions): Promise<ReactNode>;

export { type HighlightOptions, type HighlightOptionsCommon, type HighlightOptionsThemes, getHighlighter, highlight, highlightHast };
