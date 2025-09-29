import { type ClientLogEntry } from '../../shared/forward-logs-shared';
export declare const PROMISE_MARKER = "Promise {}";
export declare const UNAVAILABLE_MARKER = "[Unable to view]";
export declare const isTerminalLoggingEnabled: boolean;
/**
 * allows us to:
 * - revive the undefined log in the server as it would look in the browser
 * - not read/attempt to serialize promises (next will console error if you do that, and will cause this program to infinitely recurse)
 * - if we read a proxy that throws (no way to detect if something is a proxy), explain to the user we can't read this data
 */
export declare function preLogSerializationClone<T>(value: T, seen?: WeakMap<WeakKey, any>): any;
export declare const logStringify: (data: unknown) => string;
export declare const logQueue: {
    entries: Array<ClientLogEntry>;
    onSocketReady: (socket: WebSocket) => void;
    flushScheduled: boolean;
    socket: WebSocket | null;
    cancelFlush: (() => void) | null;
    sourceType?: 'server' | 'edge-server';
    router: 'app' | 'pages' | null;
    scheduleLogSend: (entry: ClientLogEntry) => void;
};
export declare const forwardErrorLog: (args: any[]) => void;
export declare function logUnhandledRejection(reason: unknown): void;
export declare function forwardUnhandledError(error: Error): void;
export declare const initializeDebugLogForwarding: (router: "app" | "pages") => void;
