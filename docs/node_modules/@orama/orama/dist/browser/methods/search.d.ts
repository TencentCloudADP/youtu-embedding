import { InternalDocumentID } from '../components/internal-document-id-store.js';
import type { AnyOrama, LiteralUnion, Result, Results, SearchParams, TypedDocument } from '../types.js';
export declare function search<T extends AnyOrama, ResultDocument = TypedDocument<T>>(orama: T, params: SearchParams<T, ResultDocument>, language?: string): Results<ResultDocument> | Promise<Results<ResultDocument>>;
export declare function fetchDocumentsWithDistinct<T extends AnyOrama, ResultDocument extends TypedDocument<T>>(orama: T, uniqueDocsArray: [InternalDocumentID, number][], offset: number, limit: number, distinctOn: LiteralUnion<T['schema']>): Result<ResultDocument>[];
export declare function fetchDocuments<T extends AnyOrama, ResultDocument extends TypedDocument<T>>(orama: T, uniqueDocsArray: [InternalDocumentID, number][], offset: number, limit: number): Result<ResultDocument>[];
