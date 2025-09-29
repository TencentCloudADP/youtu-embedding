import { wellKnownProperties } from '../../shared/lib/utils/reflect-utils';
const CachedSearchParams = new WeakMap();
function makeUntrackedExoticSearchParams(underlyingSearchParams) {
    const cachedSearchParams = CachedSearchParams.get(underlyingSearchParams);
    if (cachedSearchParams) {
        return cachedSearchParams;
    }
    // We don't use makeResolvedReactPromise here because searchParams
    // supports copying with spread and we don't want to unnecessarily
    // instrument the promise with spreadable properties of ReactPromise.
    const promise = Promise.resolve(underlyingSearchParams);
    CachedSearchParams.set(underlyingSearchParams, promise);
    Object.keys(underlyingSearchParams).forEach((prop)=>{
        if (wellKnownProperties.has(prop)) {
        // These properties cannot be shadowed because they need to be the
        // true underlying value for Promises to work correctly at runtime
        } else {
            ;
            promise[prop] = underlyingSearchParams[prop];
        }
    });
    return promise;
}
function makeUntrackedSearchParams(underlyingSearchParams) {
    const cachedSearchParams = CachedSearchParams.get(underlyingSearchParams);
    if (cachedSearchParams) {
        return cachedSearchParams;
    }
    const promise = Promise.resolve(underlyingSearchParams);
    CachedSearchParams.set(underlyingSearchParams, promise);
    return promise;
}
export function createRenderSearchParamsFromClient(underlyingSearchParams) {
    if (process.env.__NEXT_CACHE_COMPONENTS) {
        return makeUntrackedSearchParams(underlyingSearchParams);
    }
    return makeUntrackedExoticSearchParams(underlyingSearchParams);
}

//# sourceMappingURL=search-params.browser.prod.js.map