import type { Segment as FlightRouterStateSegment } from '../../../server/app-render/types';
type Opaque<K, T> = T & {
    __brand: K;
};
export type SegmentRequestKeyPart = Opaque<'SegmentRequestKeyPart', string>;
export type SegmentRequestKey = Opaque<'SegmentRequestKey', string>;
export type SegmentCacheKeyPart = Opaque<'SegmentCacheKeyPart', string>;
export type SegmentCacheKey = Opaque<'SegmentCacheKey', string>;
export declare const ROOT_SEGMENT_REQUEST_KEY: SegmentRequestKey;
export declare const ROOT_SEGMENT_CACHE_KEY: SegmentCacheKey;
export declare function createSegmentRequestKeyPart(segment: FlightRouterStateSegment): SegmentRequestKeyPart;
export declare function appendSegmentRequestKeyPart(parentRequestKey: SegmentRequestKey, parallelRouteKey: string, childRequestKeyPart: SegmentRequestKeyPart): SegmentRequestKey;
export declare function createSegmentCacheKeyPart(requestKeyPart: SegmentRequestKeyPart, segment: FlightRouterStateSegment): SegmentCacheKeyPart;
export declare function appendSegmentCacheKeyPart(parentSegmentKey: SegmentCacheKey, parallelRouteKey: string, childCacheKeyPart: SegmentCacheKeyPart): SegmentCacheKey;
export declare function convertSegmentPathToStaticExportFilename(segmentPath: string): string;
export {};
