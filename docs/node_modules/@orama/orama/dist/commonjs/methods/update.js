"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.update = update;
exports.updateMultiple = updateMultiple;
const hooks_js_1 = require("../components/hooks.js");
const errors_js_1 = require("../errors.js");
const insert_js_1 = require("./insert.js");
const remove_js_1 = require("./remove.js");
const utils_js_1 = require("../utils.js");
function update(orama, id, doc, language, skipHooks) {
    const asyncNeeded = (0, utils_js_1.isAsyncFunction)(orama.afterInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpdate) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpdate);
    if (asyncNeeded) {
        return updateAsync(orama, id, doc, language, skipHooks);
    }
    return updateSync(orama, id, doc, language, skipHooks);
}
async function updateAsync(orama, id, doc, language, skipHooks) {
    if (!skipHooks && orama.beforeUpdate) {
        await (0, hooks_js_1.runSingleHook)(orama.beforeUpdate, orama, id);
    }
    await (0, remove_js_1.remove)(orama, id, language, skipHooks);
    const newId = await (0, insert_js_1.insert)(orama, doc, language, skipHooks);
    if (!skipHooks && orama.afterUpdate) {
        await (0, hooks_js_1.runSingleHook)(orama.afterUpdate, orama, newId);
    }
    return newId;
}
function updateSync(orama, id, doc, language, skipHooks) {
    if (!skipHooks && orama.beforeUpdate) {
        (0, hooks_js_1.runSingleHook)(orama.beforeUpdate, orama, id);
    }
    (0, remove_js_1.remove)(orama, id, language, skipHooks);
    const newId = (0, insert_js_1.insert)(orama, doc, language, skipHooks);
    if (!skipHooks && orama.afterUpdate) {
        (0, hooks_js_1.runSingleHook)(orama.afterUpdate, orama, newId);
    }
    return newId;
}
function updateMultiple(orama, ids, docs, batchSize, language, skipHooks) {
    const asyncNeeded = (0, utils_js_1.isAsyncFunction)(orama.afterInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpdate) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpdate) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpdateMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpdateMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeRemoveMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterRemoveMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeInsertMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterInsertMultiple);
    if (asyncNeeded) {
        return updateMultipleAsync(orama, ids, docs, batchSize, language, skipHooks);
    }
    return updateMultipleSync(orama, ids, docs, batchSize, language, skipHooks);
}
async function updateMultipleAsync(orama, ids, docs, batchSize, language, skipHooks) {
    if (!skipHooks) {
        await (0, hooks_js_1.runMultipleHook)(orama.beforeUpdateMultiple, orama, ids);
    }
    const docsLength = docs.length;
    for (let i = 0; i < docsLength; i++) {
        const errorProperty = orama.validateSchema(docs[i], orama.schema);
        if (errorProperty) {
            throw (0, errors_js_1.createError)('SCHEMA_VALIDATION_FAILURE', errorProperty);
        }
    }
    await (0, remove_js_1.removeMultiple)(orama, ids, batchSize, language, skipHooks);
    const newIds = await (0, insert_js_1.innerInsertMultiple)(orama, docs, batchSize, language, skipHooks);
    if (!skipHooks) {
        await (0, hooks_js_1.runMultipleHook)(orama.afterUpdateMultiple, orama, newIds);
    }
    return newIds;
}
function updateMultipleSync(orama, ids, docs, batchSize, language, skipHooks) {
    if (!skipHooks) {
        (0, hooks_js_1.runMultipleHook)(orama.beforeUpdateMultiple, orama, ids);
    }
    const docsLength = docs.length;
    for (let i = 0; i < docsLength; i++) {
        const errorProperty = orama.validateSchema(docs[i], orama.schema);
        if (errorProperty) {
            throw (0, errors_js_1.createError)('SCHEMA_VALIDATION_FAILURE', errorProperty);
        }
    }
    (0, remove_js_1.removeMultiple)(orama, ids, batchSize, language, skipHooks);
    const newIds = (0, insert_js_1.innerInsertMultiple)(orama, docs, batchSize, language, skipHooks);
    if (!skipHooks) {
        (0, hooks_js_1.runMultipleHook)(orama.afterUpdateMultiple, orama, newIds);
    }
    return newIds;
}
//# sourceMappingURL=update.js.map