import type { ServerResponse } from 'http';
import type { NextConfigComplete } from '../config-shared';
import type { NextUrlWithParsedQuery } from '../request-meta';
export declare function isChromeDevtoolsWorkspaceUrl(url: NextUrlWithParsedQuery): boolean;
export declare function handleChromeDevtoolsWorkspaceRequest(response: ServerResponse, opts: {
    dir: string;
}, config: NextConfigComplete): Promise<void>;
