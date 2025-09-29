import { configure } from 'next/dist/compiled/safe-stable-stringify';
import { getOwnerStack, setOwnerStackIfAvailable } from './errors/stitched-error';
import { getErrorSource } from '../../../shared/lib/error-source';
import { getTerminalLoggingConfig, getIsTerminalLoggingEnabled } from './terminal-logging-config';
import { patchConsoleMethod, UNDEFINED_MARKER } from '../../shared/forward-logs-shared';
const terminalLoggingConfig = getTerminalLoggingConfig();
export const PROMISE_MARKER = 'Promise {}';
export const UNAVAILABLE_MARKER = '[Unable to view]';
const maximumDepth = typeof terminalLoggingConfig === 'object' && terminalLoggingConfig.depthLimit ? terminalLoggingConfig.depthLimit : 5;
const maximumBreadth = typeof terminalLoggingConfig === 'object' && terminalLoggingConfig.edgeLimit ? terminalLoggingConfig.edgeLimit : 100;
const stringify = configure({
    maximumDepth,
    maximumBreadth
});
export const isTerminalLoggingEnabled = getIsTerminalLoggingEnabled();
const methods = [
    'log',
    'info',
    'warn',
    'debug',
    'table',
    'assert',
    'dir',
    'dirxml',
    'group',
    'groupCollapsed',
    'groupEnd',
    'trace'
];
/**
 * allows us to:
 * - revive the undefined log in the server as it would look in the browser
 * - not read/attempt to serialize promises (next will console error if you do that, and will cause this program to infinitely recurse)
 * - if we read a proxy that throws (no way to detect if something is a proxy), explain to the user we can't read this data
 */ export function preLogSerializationClone(value, seen) {
    if (seen === void 0) seen = new WeakMap();
    if (value === undefined) return UNDEFINED_MARKER;
    if (value === null || typeof value !== 'object') return value;
    if (seen.has(value)) return seen.get(value);
    try {
        Object.keys(value);
    } catch (e) {
        return UNAVAILABLE_MARKER;
    }
    try {
        if (typeof value.then === 'function') return PROMISE_MARKER;
    } catch (e) {
        return UNAVAILABLE_MARKER;
    }
    if (Array.isArray(value)) {
        const out = [];
        seen.set(value, out);
        for (const item of value){
            try {
                out.push(preLogSerializationClone(item, seen));
            } catch (e) {
                out.push(UNAVAILABLE_MARKER);
            }
        }
        return out;
    }
    const proto = Object.getPrototypeOf(value);
    if (proto === Object.prototype || proto === null) {
        const out = {};
        seen.set(value, out);
        for (const key of Object.keys(value)){
            try {
                out[key] = preLogSerializationClone(value[key], seen);
            } catch (e) {
                out[key] = UNAVAILABLE_MARKER;
            }
        }
        return out;
    }
    return Object.prototype.toString.call(value);
}
// only safe if passed safeClone data
export const logStringify = (data)=>{
    try {
        const result = stringify(data);
        return result != null ? result : '"' + UNAVAILABLE_MARKER + '"';
    } catch (e) {
        return '"' + UNAVAILABLE_MARKER + '"';
    }
};
const afterThisFrame = (cb)=>{
    let timeout;
    const rafId = requestAnimationFrame(()=>{
        timeout = setTimeout(()=>{
            cb();
        });
    });
    return ()=>{
        cancelAnimationFrame(rafId);
        clearTimeout(timeout);
    };
};
let isPatched = false;
const serializeEntries = (entries)=>entries.map((clientEntry)=>{
        switch(clientEntry.kind){
            case 'any-logged-error':
            case 'console':
                {
                    return {
                        ...clientEntry,
                        args: clientEntry.args.map(stringifyUserArg)
                    };
                }
            case 'formatted-error':
                {
                    return clientEntry;
                }
            default:
                {
                    return null;
                }
        }
    });
export const logQueue = {
    entries: [],
    flushScheduled: false,
    cancelFlush: null,
    socket: null,
    sourceType: undefined,
    router: null,
    scheduleLogSend: (entry)=>{
        logQueue.entries.push(entry);
        if (logQueue.flushScheduled) {
            return;
        }
        // safe to deref and use in setTimeout closure since we cancel on new socket
        const socket = logQueue.socket;
        if (!socket) {
            return;
        }
        // we probably dont need this
        logQueue.flushScheduled = true;
        // non blocking log flush, runs at most once per frame
        logQueue.cancelFlush = afterThisFrame(()=>{
            logQueue.flushScheduled = false;
            // just incase
            try {
                const payload = JSON.stringify({
                    event: 'browser-logs',
                    entries: serializeEntries(logQueue.entries),
                    router: logQueue.router,
                    // needed for source mapping, we just assign the sourceType from the last error for the whole batch
                    sourceType: logQueue.sourceType
                });
                socket.send(payload);
                logQueue.entries = [];
                logQueue.sourceType = undefined;
            } catch (e) {
            // error (make sure u don't infinite loop)
            /* noop */ }
        });
    },
    onSocketReady: (socket)=>{
        if (socket.readyState !== WebSocket.OPEN) {
            // invariant
            return;
        }
        // incase an existing timeout was going to run with a stale socket
        logQueue.cancelFlush == null ? void 0 : logQueue.cancelFlush.call(logQueue);
        logQueue.socket = socket;
        try {
            const payload = JSON.stringify({
                event: 'browser-logs',
                entries: serializeEntries(logQueue.entries),
                router: logQueue.router,
                sourceType: logQueue.sourceType
            });
            socket.send(payload);
            logQueue.entries = [];
            logQueue.sourceType = undefined;
        } catch (e) {
        /** noop just incase */ }
    }
};
const stringifyUserArg = (arg)=>{
    if (arg.kind !== 'arg') {
        return arg;
    }
    return {
        ...arg,
        data: logStringify(arg.data)
    };
};
const createErrorArg = (error)=>{
    const stack = stackWithOwners(error);
    return {
        kind: 'formatted-error-arg',
        prefix: error.message ? error.name + ": " + error.message : "" + error.name,
        stack
    };
};
const createLogEntry = (level, args)=>{
    // do not abstract this, it implicitly relies on which functions call it. forcing the inlined implementation makes you think about callers
    // error capture stack trace maybe
    const stack = stackWithOwners(new Error());
    const stackLines = stack == null ? void 0 : stack.split('\n');
    const cleanStack = stackLines == null ? void 0 : stackLines.slice(3).join('\n') // this is probably ignored anyways
    ;
    const entry = {
        kind: 'console',
        consoleMethodStack: cleanStack != null ? cleanStack : null,
        method: level,
        args: args.map((arg)=>{
            if (arg instanceof Error) {
                return createErrorArg(arg);
            }
            return {
                kind: 'arg',
                data: preLogSerializationClone(arg)
            };
        })
    };
    logQueue.scheduleLogSend(entry);
};
export const forwardErrorLog = (args)=>{
    const errorObjects = args.filter((arg)=>arg instanceof Error);
    const first = errorObjects.at(0);
    if (first) {
        const source = getErrorSource(first);
        if (source) {
            logQueue.sourceType = source;
        }
    }
    /**
   * browser shows stack regardless of type of data passed to console.error, so we should do the same
   *
   * do not abstract this, it implicitly relies on which functions call it. forcing the inlined implementation makes you think about callers
   */ const stack = stackWithOwners(new Error());
    const stackLines = stack == null ? void 0 : stack.split('\n');
    const cleanStack = stackLines == null ? void 0 : stackLines.slice(3).join('\n');
    const entry = {
        kind: 'any-logged-error',
        method: 'error',
        consoleErrorStack: cleanStack != null ? cleanStack : '',
        args: args.map((arg)=>{
            if (arg instanceof Error) {
                return createErrorArg(arg);
            }
            return {
                kind: 'arg',
                data: preLogSerializationClone(arg)
            };
        })
    };
    logQueue.scheduleLogSend(entry);
};
const createUncaughtErrorEntry = (errorName, errorMessage, fullStack)=>{
    const entry = {
        kind: 'formatted-error',
        prefix: "Uncaught " + errorName + ": " + errorMessage,
        stack: fullStack,
        method: 'error'
    };
    logQueue.scheduleLogSend(entry);
};
const stackWithOwners = (error)=>{
    let ownerStack = '';
    setOwnerStackIfAvailable(error);
    ownerStack = getOwnerStack(error) || '';
    const stack = (error.stack || '') + ownerStack;
    return stack;
};
export function logUnhandledRejection(reason) {
    if (reason instanceof Error) {
        createUnhandledRejectionErrorEntry(reason, stackWithOwners(reason));
        return;
    }
    createUnhandledRejectionNonErrorEntry(reason);
}
const createUnhandledRejectionErrorEntry = (error, fullStack)=>{
    const source = getErrorSource(error);
    if (source) {
        logQueue.sourceType = source;
    }
    const entry = {
        kind: 'formatted-error',
        prefix: "⨯ unhandledRejection: " + error.name + ": " + error.message,
        stack: fullStack,
        method: 'error'
    };
    logQueue.scheduleLogSend(entry);
};
const createUnhandledRejectionNonErrorEntry = (reason)=>{
    const entry = {
        kind: 'any-logged-error',
        // we can't access the stack since the event is dispatched async and creating an inline error would be meaningless
        consoleErrorStack: '',
        method: 'error',
        args: [
            {
                kind: 'arg',
                data: "⨯ unhandledRejection:",
                isRejectionMessage: true
            },
            {
                kind: 'arg',
                data: preLogSerializationClone(reason)
            }
        ]
    };
    logQueue.scheduleLogSend(entry);
};
const isHMR = (args)=>{
    const firstArg = args[0];
    if (typeof firstArg !== 'string') {
        return false;
    }
    if (firstArg.startsWith('[Fast Refresh]')) {
        return true;
    }
    if (firstArg.startsWith('[HMR]')) {
        return true;
    }
    return false;
};
const isIgnoredLog = (args)=>{
    if (args.length < 3) {
        return false;
    }
    const [format, styles, label] = args;
    if (typeof format !== 'string' || typeof styles !== 'string' || typeof label !== 'string') {
        return false;
    }
    // kinda hacky, we should define a common format for these strings so we can safely ignore
    return format.startsWith('%c%s%c') && styles.includes('background:');
};
export function forwardUnhandledError(error) {
    createUncaughtErrorEntry(error.name, error.message, stackWithOwners(error));
}
// TODO: this router check is brittle, we need to update based on the current router the user is using
export const initializeDebugLogForwarding = (router)=>{
    // probably don't need this
    if (isPatched) {
        return;
    }
    // TODO(rob): why does this break rendering on server, important to know incase the same bug appears in browser
    if (typeof window === 'undefined') {
        return;
    }
    // better to be safe than sorry
    try {
        methods.forEach((method)=>patchConsoleMethod(method, function(_) {
                for(var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++){
                    args[_key - 1] = arguments[_key];
                }
                if (isHMR(args)) {
                    return;
                }
                if (isIgnoredLog(args)) {
                    return;
                }
                createLogEntry(method, args);
            }));
    } catch (e) {}
    logQueue.router = router;
    isPatched = true;
};

//# sourceMappingURL=forward-logs.js.map