import type { IncomingMessage, ServerResponse } from 'node:http';
import type { NextConfigComplete } from '../../config-shared';
import type { UrlWithParsedQuery } from 'node:url';
export type RevalidateFn = (config: {
    urlPath: string;
    revalidateHeaders: {
        [key: string]: string | string[];
    };
    opts: {
        unstable_onlyGenerated?: boolean;
    };
}) => Promise<void>;
export type RouterServerContext = Record<string, {
    hostname?: string;
    revalidate?: RevalidateFn;
    render404?: (req: IncomingMessage, res: ServerResponse, parsedUrl?: UrlWithParsedQuery, setHeaders?: boolean) => Promise<void>;
    publicRuntimeConfig?: NextConfigComplete['publicRuntimeConfig'];
    nextConfig?: NextConfigComplete;
    isCustomServer?: boolean;
    experimentalTestProxy?: boolean;
    logErrorWithOriginalStack?: (err: unknown, type: string) => void;
    setIsrStatus?: (key: string, value: boolean | null) => void;
}>;
export declare const RouterServerContextSymbol: unique symbol;
export declare const routerServerGlobal: typeof globalThis & {
    [RouterServerContextSymbol]?: RouterServerContext;
};
