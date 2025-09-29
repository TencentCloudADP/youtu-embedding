import type { HMR_ACTION_TYPES } from '../../../server/dev/hot-reloader-types';
export declare const REACT_REFRESH_FULL_RELOAD: string;
export declare const REACT_REFRESH_FULL_RELOAD_FROM_ERROR = "[Fast Refresh] performing full reload because your application had an unrecoverable error";
export declare function reportInvalidHmrMessage(message: HMR_ACTION_TYPES | MessageEvent<unknown>, err: unknown): void;
