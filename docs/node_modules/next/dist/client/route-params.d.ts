import type { DynamicParamTypesShort } from '../server/app-render/types';
import type { NormalizedSearch } from './components/segment-cache';
import type { RSCResponse } from './components/router-reducer/fetch-server-response';
export type RouteParamValue = string | Array<string> | null;
export type RouteParam = {
    name: string;
    value: RouteParamValue;
    type: DynamicParamTypesShort;
};
export declare function getRenderedSearch(response: RSCResponse): NormalizedSearch;
export declare function getRenderedPathname(response: RSCResponse): string;
export declare function parseDynamicParamFromURLPart(paramType: DynamicParamTypesShort, pathnameParts: Array<string>, partIndex: number): RouteParamValue;
export declare function doesStaticSegmentAppearInURL(segment: string): boolean;
export declare function getCacheKeyForDynamicParam(paramValue: RouteParamValue, renderedSearch: NormalizedSearch): string;
export declare function urlToUrlWithoutFlightMarker(url: URL): URL;
export declare function getParamValueFromCacheKey(paramCacheKey: string, paramType: DynamicParamTypesShort): string | string[];
