import type { ExperimentalConfig, NextConfigComplete, NextConfig } from './config-shared';
export { normalizeConfig } from './config-shared';
export type { DomainLocale, NextConfig } from './config-shared';
export declare function warnOptionHasBeenDeprecated(config: NextConfig, nestedPropertyKey: string, reason: string, silent: boolean): boolean;
export declare function warnOptionHasBeenMovedOutOfExperimental(config: NextConfig, oldExperimentalKey: string, newKey: string, configFileName: string, silent: boolean): NextConfig;
export default function loadConfig(phase: string, dir: string, { customConfig, rawConfig, silent, reportExperimentalFeatures, reactProductionProfiling, debugPrerender, }?: {
    customConfig?: object | null;
    rawConfig?: boolean;
    silent?: boolean;
    reportExperimentalFeatures?: (configuredExperimentalFeatures: ConfiguredExperimentalFeature[]) => void;
    reactProductionProfiling?: boolean;
    debugPrerender?: boolean;
}): Promise<NextConfigComplete>;
export type ConfiguredExperimentalFeature = {
    key: keyof ExperimentalConfig;
    value: ExperimentalConfig[keyof ExperimentalConfig];
    reason?: string;
};
