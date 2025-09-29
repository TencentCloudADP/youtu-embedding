import type { CacheNodeSeedData, FlightData, FlightDataPath, FlightRouterState, FlightSegmentPath, Segment } from '../server/app-render/types';
import type { HeadData } from '../shared/lib/app-router-context.shared-runtime';
export type NormalizedFlightData = {
    /**
     * The full `FlightSegmentPath` inclusive of the final `Segment`
     */
    segmentPath: FlightSegmentPath;
    /**
     * The `FlightSegmentPath` exclusive of the final `Segment`
     */
    pathToSegment: FlightSegmentPath;
    segment: Segment;
    tree: FlightRouterState;
    seedData: CacheNodeSeedData | null;
    head: HeadData;
    isHeadPartial: boolean;
    isRootRender: boolean;
};
export declare function getFlightDataPartsFromPath(flightDataPath: FlightDataPath): NormalizedFlightData;
export declare function getNextFlightSegmentPath(flightSegmentPath: FlightSegmentPath): FlightSegmentPath;
export declare function normalizeFlightData(flightData: FlightData): NormalizedFlightData[] | string;
/**
 * This function is used to prepare the flight router state for the request.
 * It removes markers that are not needed by the server, and are purely used
 * for stashing state on the client.
 * @param flightRouterState - The flight router state to prepare.
 * @param isHmrRefresh - Whether this is an HMR refresh request.
 * @returns The prepared flight router state.
 */
export declare function prepareFlightRouterStateForRequest(flightRouterState: FlightRouterState, isHmrRefresh?: boolean): string;
