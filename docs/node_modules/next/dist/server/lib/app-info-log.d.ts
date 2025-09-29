import { type ConfiguredExperimentalFeature } from '../config';
export declare function logStartInfo({ networkUrl, appUrl, envInfo, experimentalFeatures, maxExperimentalFeatures, }: {
    networkUrl: string | null;
    appUrl: string | null;
    envInfo?: string[];
    experimentalFeatures?: ConfiguredExperimentalFeature[];
    maxExperimentalFeatures?: number;
}): void;
export declare function getStartServerInfo({ dir, dev, debugPrerender, }: {
    dir: string;
    dev: boolean;
    debugPrerender?: boolean;
}): Promise<{
    envInfo?: string[];
    experimentalFeatures?: ConfiguredExperimentalFeature[];
}>;
