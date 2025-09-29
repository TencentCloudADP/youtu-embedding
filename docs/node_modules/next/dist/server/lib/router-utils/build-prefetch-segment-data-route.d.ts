export declare const SEGMENT_PATH_KEY = "nextSegmentPath";
export type PrefetchSegmentDataRoute = {
    source: string;
    destination: string;
    routeKeys: {
        [key: string]: string;
    };
};
export declare function buildPrefetchSegmentDataRoute(page: string, segmentPath: string): PrefetchSegmentDataRoute;
/**
 * Builds a prefetch segment data route that is inverted. This means that it's
 * supposed to rewrite from the previous segment paths route back to the
 * prefetch RSC route.
 *
 * @param page - The page to build the route for.
 * @param segmentPath - The segment path to build the route for.
 * @returns The prefetch segment data route.
 */
export declare function buildInversePrefetchSegmentDataRoute(page: string, segmentPath: string): PrefetchSegmentDataRoute;
