import { HighlighterGeneric } from '@shikijs/types';
import { Root } from 'hast';
import { Transformer } from 'unified';
import { R as RehypeShikiCoreOptions } from './shared/rehype.DcMmi29I.mjs';
export { M as MapLike, a as RehypeShikiExtraOptions } from './shared/rehype.DcMmi29I.mjs';
import 'shiki';

declare function rehypeShikiFromHighlighter(highlighter: HighlighterGeneric<any, any>, options: RehypeShikiCoreOptions): Transformer<Root, Root>;

export { RehypeShikiCoreOptions, rehypeShikiFromHighlighter as default };
