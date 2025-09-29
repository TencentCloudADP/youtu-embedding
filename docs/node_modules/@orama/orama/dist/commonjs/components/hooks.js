"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SINGLE_OR_ARRAY_COMPONENTS = exports.FUNCTION_COMPONENTS = exports.OBJECT_COMPONENTS = void 0;
exports.runSingleHook = runSingleHook;
exports.runMultipleHook = runMultipleHook;
exports.runAfterSearch = runAfterSearch;
exports.runBeforeSearch = runBeforeSearch;
exports.runAfterCreate = runAfterCreate;
const utils_js_1 = require("../utils.js");
exports.OBJECT_COMPONENTS = ['tokenizer', 'index', 'documentsStore', 'sorter'];
exports.FUNCTION_COMPONENTS = [
    'validateSchema',
    'getDocumentIndexId',
    'getDocumentProperties',
    'formatElapsedTime'
];
exports.SINGLE_OR_ARRAY_COMPONENTS = [
/* deprecated with v2.0.0-beta.5 */
];
function runSingleHook(hooks, orama, id, doc) {
    const needAsync = hooks.some(utils_js_1.isAsyncFunction);
    if (needAsync) {
        return (async () => {
            for (const hook of hooks) {
                await hook(orama, id, doc);
            }
        })();
    }
    else {
        for (const hook of hooks) {
            hook(orama, id, doc);
        }
    }
}
function runMultipleHook(hooks, orama, docsOrIds) {
    const needAsync = hooks.some(utils_js_1.isAsyncFunction);
    if (needAsync) {
        return (async () => {
            for (const hook of hooks) {
                await hook(orama, docsOrIds);
            }
        })();
    }
    else {
        for (const hook of hooks) {
            hook(orama, docsOrIds);
        }
    }
}
function runAfterSearch(hooks, db, params, language, results) {
    const needAsync = hooks.some(utils_js_1.isAsyncFunction);
    if (needAsync) {
        return (async () => {
            for (const hook of hooks) {
                await hook(db, params, language, results);
            }
        })();
    }
    else {
        for (const hook of hooks) {
            hook(db, params, language, results);
        }
    }
}
function runBeforeSearch(hooks, db, params, language) {
    const needAsync = hooks.some(utils_js_1.isAsyncFunction);
    if (needAsync) {
        return (async () => {
            for (const hook of hooks) {
                await hook(db, params, language);
            }
        })();
    }
    else {
        for (const hook of hooks) {
            hook(db, params, language);
        }
    }
}
function runAfterCreate(hooks, db) {
    const needAsync = hooks.some(utils_js_1.isAsyncFunction);
    if (needAsync) {
        return (async () => {
            for (const hook of hooks) {
                await hook(db);
            }
        })();
    }
    else {
        for (const hook of hooks) {
            hook(db);
        }
    }
}
//# sourceMappingURL=hooks.js.map