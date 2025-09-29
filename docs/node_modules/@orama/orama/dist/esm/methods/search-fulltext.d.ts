import { Language } from '../components/tokenizer/languages.js';
import type { AnyOrama, BM25Params, Results, SearchParamsFullText, TokenScore, TypedDocument } from '../types.js';
export declare function innerFullTextSearch<T extends AnyOrama>(orama: T, params: Pick<SearchParamsFullText<T>, 'term' | 'properties' | 'where' | 'exact' | 'tolerance' | 'boost' | 'relevance' | 'threshold'>, language: Language | undefined): TokenScore[];
export declare function fullTextSearch<T extends AnyOrama, ResultDocument = TypedDocument<T>>(orama: T, params: SearchParamsFullText<T, ResultDocument>, language?: string): Results<ResultDocument> | Promise<Results<ResultDocument>>;
export declare const defaultBM25Params: BM25Params;
