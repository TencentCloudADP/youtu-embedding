"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// Web platforms don't have process. React-Native doesn't have process.emitWarning.
const warn = globalThis.process?.emitWarning ??
    function emitWarning(message, options) {
        console.warn(`[WARNING] [${options.code}] ${message}`);
    };
//# sourceMappingURL=sync-blocking-checker.js.map