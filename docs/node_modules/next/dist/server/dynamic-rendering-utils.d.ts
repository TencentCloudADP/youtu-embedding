export declare function isHangingPromiseRejectionError(err: unknown): err is HangingPromiseRejectionError;
declare class HangingPromiseRejectionError extends Error {
    readonly route: string;
    readonly expression: string;
    readonly digest = "HANGING_PROMISE_REJECTION";
    constructor(route: string, expression: string);
}
export declare function makeDevtoolsIOAwarePromise<T>(underlying: T): Promise<T>;
export {};
