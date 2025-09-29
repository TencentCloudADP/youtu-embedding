import { Span } from '../trace';
import type { NextConfigComplete } from '../server/config-shared';
import { type BuildTraceContext } from './webpack/plugins/next-trace-entrypoints-plugin';
import type { RoutesUsingEdgeRuntime } from './utils';
export declare function collectBuildTraces({ dir, config, distDir, edgeRuntimeRoutes, staticPages, nextBuildSpan, hasSsrAmpPages, buildTraceContext, outputFileTracingRoot, isTurbopack, }: {
    dir: string;
    distDir: string;
    staticPages: string[];
    hasSsrAmpPages: boolean;
    outputFileTracingRoot: string;
    edgeRuntimeRoutes: RoutesUsingEdgeRuntime;
    nextBuildSpan?: Span;
    config: NextConfigComplete;
    buildTraceContext?: BuildTraceContext;
    isTurbopack: boolean;
}): Promise<void>;
