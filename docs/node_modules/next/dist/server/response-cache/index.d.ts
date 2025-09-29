import type { ResponseCacheEntry, ResponseGenerator, ResponseCacheBase, IncrementalResponseCache } from './types';
import type { RouteKind } from '../route-kind';
export * from './types';
export default class ResponseCache implements ResponseCacheBase {
    private readonly batcher;
    private previousCacheItem?;
    private minimal_mode?;
    constructor(minimal_mode: boolean);
    get(key: string | null, responseGenerator: ResponseGenerator, context: {
        routeKind: RouteKind;
        isOnDemandRevalidate?: boolean;
        isPrefetch?: boolean;
        incrementalCache: IncrementalResponseCache;
        isRoutePPREnabled?: boolean;
        isFallback?: boolean;
        waitUntil?: (prom: Promise<any>) => void;
    }): Promise<ResponseCacheEntry | null>;
}
