import type { LoaderTree } from '../lib/app-dir-module';
export declare function parseLoaderTree(tree: LoaderTree): {
    page: import("../../build/webpack/loaders/metadata/types").ModuleTuple | undefined;
    segment: string;
    modules: import("../../build/webpack/loaders/next-app-loader").AppDirModules;
    conventionPath: string | undefined;
    parallelRoutes: {
        [parallelRouterKey: string]: LoaderTree;
    };
};
