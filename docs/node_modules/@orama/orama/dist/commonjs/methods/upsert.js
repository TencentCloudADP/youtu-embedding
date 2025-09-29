"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.upsert = upsert;
exports.upsertMultiple = upsertMultiple;
const hooks_js_1 = require("../components/hooks.js");
const errors_js_1 = require("../errors.js");
const insert_js_1 = require("./insert.js");
const update_js_1 = require("./update.js");
const utils_js_1 = require("../utils.js");
function upsert(orama, doc, language, skipHooks, options) {
    const asyncNeeded = (0, utils_js_1.isAsyncFunction)(orama.afterInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpdate) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpdate) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.insert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.afterInsert);
    if (asyncNeeded) {
        return upsertAsync(orama, doc, language, skipHooks, options);
    }
    return upsertSync(orama, doc, language, skipHooks, options);
}
async function upsertAsync(orama, doc, language, skipHooks, options) {
    const id = orama.getDocumentIndexId(doc);
    if (typeof id !== 'string') {
        throw (0, errors_js_1.createError)('DOCUMENT_ID_MUST_BE_STRING', typeof id);
    }
    if (!skipHooks && orama.beforeUpsert) {
        await (0, hooks_js_1.runSingleHook)(orama.beforeUpsert, orama, id, doc);
    }
    // Check if document exists
    const existingDoc = orama.documentsStore.get(orama.data.docs, id);
    let resultId;
    if (existingDoc) {
        // Document exists, update it
        resultId = await (0, update_js_1.update)(orama, id, doc, language, skipHooks);
    }
    else {
        // Document doesn't exist, insert it
        resultId = await (0, insert_js_1.insert)(orama, doc, language, skipHooks, options);
    }
    if (!skipHooks && orama.afterUpsert) {
        await (0, hooks_js_1.runSingleHook)(orama.afterUpsert, orama, resultId, doc);
    }
    return resultId;
}
function upsertSync(orama, doc, language, skipHooks, options) {
    const id = orama.getDocumentIndexId(doc);
    if (typeof id !== 'string') {
        throw (0, errors_js_1.createError)('DOCUMENT_ID_MUST_BE_STRING', typeof id);
    }
    if (!skipHooks && orama.beforeUpsert) {
        (0, hooks_js_1.runSingleHook)(orama.beforeUpsert, orama, id, doc);
    }
    // Check if document exists
    const existingDoc = orama.documentsStore.get(orama.data.docs, id);
    let resultId;
    if (existingDoc) {
        // Document exists, update it
        resultId = (0, update_js_1.update)(orama, id, doc, language, skipHooks);
    }
    else {
        // Document doesn't exist, insert it
        resultId = (0, insert_js_1.insert)(orama, doc, language, skipHooks, options);
    }
    if (!skipHooks && orama.afterUpsert) {
        (0, hooks_js_1.runSingleHook)(orama.afterUpsert, orama, resultId, doc);
    }
    return resultId;
}
function upsertMultiple(orama, docs, batchSize, language, skipHooks) {
    const asyncNeeded = (0, utils_js_1.isAsyncFunction)(orama.afterInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeRemove) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpdate) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpdate) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpsertMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpsertMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeInsertMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterInsertMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeUpdateMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterUpdateMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeRemoveMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterRemoveMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.insert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.afterInsert);
    if (asyncNeeded) {
        return upsertMultipleAsync(orama, docs, batchSize, language, skipHooks);
    }
    return upsertMultipleSync(orama, docs, batchSize, language, skipHooks);
}
async function upsertMultipleAsync(orama, docs, batchSize, language, skipHooks) {
    if (!skipHooks && orama.beforeUpsertMultiple) {
        await (0, hooks_js_1.runMultipleHook)(orama.beforeUpsertMultiple, orama, docs);
    }
    // Validate all documents first
    const docsLength = docs.length;
    for (let i = 0; i < docsLength; i++) {
        const errorProperty = orama.validateSchema(docs[i], orama.schema);
        if (errorProperty) {
            throw (0, errors_js_1.createError)('SCHEMA_VALIDATION_FAILURE', errorProperty);
        }
    }
    // Separate documents into insert and update arrays
    const docsToInsert = [];
    const docsToUpdate = [];
    const idsToUpdate = [];
    for (const doc of docs) {
        const id = orama.getDocumentIndexId(doc);
        if (typeof id !== 'string') {
            throw (0, errors_js_1.createError)('DOCUMENT_ID_MUST_BE_STRING', typeof id);
        }
        const existingDoc = orama.documentsStore.get(orama.data.docs, id);
        if (existingDoc) {
            docsToUpdate.push(doc);
            idsToUpdate.push(id);
        }
        else {
            docsToInsert.push(doc);
        }
    }
    // Perform bulk operations
    const results = [];
    if (docsToUpdate.length > 0) {
        const updateResults = await (0, update_js_1.updateMultiple)(orama, idsToUpdate, docsToUpdate, batchSize, language, skipHooks);
        results.push(...updateResults);
    }
    if (docsToInsert.length > 0) {
        const insertResults = await (0, insert_js_1.innerInsertMultiple)(orama, docsToInsert, batchSize, language, skipHooks);
        results.push(...insertResults);
    }
    if (!skipHooks && orama.afterUpsertMultiple) {
        await (0, hooks_js_1.runMultipleHook)(orama.afterUpsertMultiple, orama, results);
    }
    return results;
}
function upsertMultipleSync(orama, docs, batchSize, language, skipHooks) {
    if (!skipHooks && orama.beforeUpsertMultiple) {
        (0, hooks_js_1.runMultipleHook)(orama.beforeUpsertMultiple, orama, docs);
    }
    // Validate all documents first
    const docsLength = docs.length;
    for (let i = 0; i < docsLength; i++) {
        const errorProperty = orama.validateSchema(docs[i], orama.schema);
        if (errorProperty) {
            throw (0, errors_js_1.createError)('SCHEMA_VALIDATION_FAILURE', errorProperty);
        }
    }
    // Separate documents into insert and update arrays
    const docsToInsert = [];
    const docsToUpdate = [];
    const idsToUpdate = [];
    for (const doc of docs) {
        const id = orama.getDocumentIndexId(doc);
        if (typeof id !== 'string') {
            throw (0, errors_js_1.createError)('DOCUMENT_ID_MUST_BE_STRING', typeof id);
        }
        const existingDoc = orama.documentsStore.get(orama.data.docs, id);
        if (existingDoc) {
            docsToUpdate.push(doc);
            idsToUpdate.push(id);
        }
        else {
            docsToInsert.push(doc);
        }
    }
    // Perform bulk operations
    const results = [];
    if (docsToUpdate.length > 0) {
        const updateResults = (0, update_js_1.updateMultiple)(orama, idsToUpdate, docsToUpdate, batchSize, language, skipHooks);
        results.push(...updateResults);
    }
    if (docsToInsert.length > 0) {
        const insertResults = (0, insert_js_1.innerInsertMultiple)(orama, docsToInsert, batchSize, language, skipHooks);
        results.push(...insertResults);
    }
    if (!skipHooks && orama.afterUpsertMultiple) {
        (0, hooks_js_1.runMultipleHook)(orama.afterUpsertMultiple, orama, results);
    }
    return results;
}
//# sourceMappingURL=upsert.js.map