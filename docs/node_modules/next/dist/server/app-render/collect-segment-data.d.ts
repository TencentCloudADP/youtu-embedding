import type { DynamicParamTypesShort } from './types';
import type { ManifestNode } from '../../build/webpack/plugins/flight-manifest-plugin';
import type { HeadData, LoadingModuleData } from '../../shared/lib/app-router-context.shared-runtime';
import { type SegmentRequestKey } from '../../shared/lib/segment-cache/segment-value-encoding';
export type RootTreePrefetch = {
    buildId: string;
    tree: TreePrefetch;
    head: HeadData;
    isHeadPartial: boolean;
    staleTime: number;
};
export type TreePrefetch = {
    name: string;
    paramType: DynamicParamTypesShort | null;
    paramKey: string | null;
    slots: null | {
        [parallelRouteKey: string]: TreePrefetch;
    };
    isRootLayout: boolean;
};
export type SegmentPrefetch = {
    buildId: string;
    rsc: React.ReactNode | null;
    loading: LoadingModuleData | Promise<LoadingModuleData>;
    isPartial: boolean;
};
export declare function collectSegmentData(isClientParamParsingEnabled: boolean, fullPageDataBuffer: Buffer, staleTime: number, clientModules: ManifestNode, serverConsumerManifest: any): Promise<Map<SegmentRequestKey, Buffer>>;
