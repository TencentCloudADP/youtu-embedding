import type { LoaderTree } from '../../server/lib/app-dir-module';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { AppPageRouteModule } from '../../server/route-modules/app-page/module.compiled';
/**
 * The tree created in next-app-loader that holds component segments and modules
 * and I've updated it.
 */
declare const tree: LoaderTree;
declare const pages: any;
export { tree, pages };
import GlobalError from 'VAR_MODULE_GLOBAL_ERROR';
export { GlobalError };
export declare const __next_app__: {
    require: (id: string | number) => unknown;
    loadChunk: (id: string | number) => Promise<unknown>;
};
export * from '../../server/app-render/entry-base';
export declare const routeModule: AppPageRouteModule;
export declare function handler(req: IncomingMessage, res: ServerResponse, ctx: {
    waitUntil: (prom: Promise<void>) => void;
}): Promise<void | null>;
