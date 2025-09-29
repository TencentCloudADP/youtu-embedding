import type { FunctionsConfigManifest, PrerenderManifest, RoutesManifest } from '..';
import type { MiddlewareManifest } from '../webpack/plugins/middleware-plugin';
export declare function handleBuildComplete({ distDir, tracingRoot, adapterPath, pageKeys, appPageKeys, hasNodeMiddleware, hasInstrumentationHook, requiredServerFiles, routesManifest, prerenderManifest, middlewareManifest, functionsConfigManifest, hasStatic404, }: {
    dir: string;
    distDir: string;
    adapterPath: string;
    tracingRoot: string;
    hasNodeMiddleware: boolean;
    pageKeys: readonly string[];
    hasInstrumentationHook: boolean;
    appPageKeys?: readonly string[] | undefined;
    requiredServerFiles: string[];
    routesManifest: RoutesManifest;
    prerenderManifest: PrerenderManifest;
    middlewareManifest: MiddlewareManifest;
    functionsConfigManifest: FunctionsConfigManifest;
    hasStatic404: boolean;
}): Promise<void>;
