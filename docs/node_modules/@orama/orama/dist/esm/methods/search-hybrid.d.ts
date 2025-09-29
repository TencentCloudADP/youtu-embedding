import type { AnyOrama, TypedDocument, SearchParamsHybrid, Results } from '../types.js';
export declare function innerHybridSearch<T extends AnyOrama, ResultDocument = TypedDocument<T>>(orama: T, params: SearchParamsHybrid<T, ResultDocument>, language?: string): [any, any][];
export declare function hybridSearch<T extends AnyOrama, ResultDocument = TypedDocument<T>>(orama: T, params: SearchParamsHybrid<T, ResultDocument>, language?: string): Results<ResultDocument> | Promise<Results<ResultDocument>>;
