import type { FlightRouterState, FlightSegmentPath } from '../../../../server/app-render/types';
import { type Mutable, type NavigateAction, type ReadonlyReducerState, type ReducerState } from '../router-reducer-types';
export declare function handleExternalUrl(state: ReadonlyReducerState, mutable: Mutable, url: string, pendingPush: boolean): ReducerState;
export declare function generateSegmentsFromPatch(flightRouterPatch: FlightRouterState): FlightSegmentPath[];
export declare function navigateReducer(state: ReadonlyReducerState, action: NavigateAction): ReducerState;
