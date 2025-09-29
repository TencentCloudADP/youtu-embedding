"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.insert = insert;
exports.insertMultiple = insertMultiple;
exports.innerInsertMultiple = innerInsertMultiple;
const components_js_1 = require("../components.js");
const utils_js_1 = require("../utils.js");
const hooks_js_1 = require("../components/hooks.js");
const errors_js_1 = require("../errors.js");
const internal_document_id_store_js_1 = require("../components/internal-document-id-store.js");
function insert(orama, doc, language, skipHooks, options) {
    const errorProperty = orama.validateSchema(doc, orama.schema);
    if (errorProperty) {
        throw (0, errors_js_1.createError)('SCHEMA_VALIDATION_FAILURE', errorProperty);
    }
    const asyncNeeded = (0, utils_js_1.isAsyncFunction)(orama.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.insert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.afterInsert);
    if (asyncNeeded) {
        return innerInsertAsync(orama, doc, language, skipHooks, options);
    }
    return innerInsertSync(orama, doc, language, skipHooks, options);
}
const ENUM_TYPE = new Set(['enum', 'enum[]']);
const STRING_NUMBER_TYPE = new Set(['string', 'number']);
async function innerInsertAsync(orama, doc, language, skipHooks, options) {
    const { index, docs } = orama.data;
    const id = orama.getDocumentIndexId(doc);
    if (typeof id !== 'string') {
        throw (0, errors_js_1.createError)('DOCUMENT_ID_MUST_BE_STRING', typeof id);
    }
    const internalId = (0, internal_document_id_store_js_1.getInternalDocumentId)(orama.internalDocumentIDStore, id);
    if (!skipHooks) {
        await (0, hooks_js_1.runSingleHook)(orama.beforeInsert, orama, id, doc);
    }
    if (!orama.documentsStore.store(docs, id, internalId, doc)) {
        throw (0, errors_js_1.createError)('DOCUMENT_ALREADY_EXISTS', id);
    }
    const docsCount = orama.documentsStore.count(docs);
    const indexableProperties = orama.index.getSearchableProperties(index);
    const indexablePropertiesWithTypes = orama.index.getSearchablePropertiesWithTypes(index);
    const indexableValues = orama.getDocumentProperties(doc, indexableProperties);
    for (const [key, value] of Object.entries(indexableValues)) {
        if (typeof value === 'undefined')
            continue;
        const actualType = typeof value;
        const expectedType = indexablePropertiesWithTypes[key];
        validateDocumentProperty(actualType, expectedType, key, value);
    }
    await indexAndSortDocument(orama, id, indexableProperties, indexableValues, docsCount, language, doc, options);
    if (!skipHooks) {
        await (0, hooks_js_1.runSingleHook)(orama.afterInsert, orama, id, doc);
    }
    return id;
}
function innerInsertSync(orama, doc, language, skipHooks, options) {
    const { index, docs } = orama.data;
    const id = orama.getDocumentIndexId(doc);
    if (typeof id !== 'string') {
        throw (0, errors_js_1.createError)('DOCUMENT_ID_MUST_BE_STRING', typeof id);
    }
    const internalId = (0, internal_document_id_store_js_1.getInternalDocumentId)(orama.internalDocumentIDStore, id);
    if (!skipHooks) {
        (0, hooks_js_1.runSingleHook)(orama.beforeInsert, orama, id, doc);
    }
    if (!orama.documentsStore.store(docs, id, internalId, doc)) {
        throw (0, errors_js_1.createError)('DOCUMENT_ALREADY_EXISTS', id);
    }
    const docsCount = orama.documentsStore.count(docs);
    const indexableProperties = orama.index.getSearchableProperties(index);
    const indexablePropertiesWithTypes = orama.index.getSearchablePropertiesWithTypes(index);
    const indexableValues = orama.getDocumentProperties(doc, indexableProperties);
    for (const [key, value] of Object.entries(indexableValues)) {
        if (typeof value === 'undefined')
            continue;
        const actualType = typeof value;
        const expectedType = indexablePropertiesWithTypes[key];
        validateDocumentProperty(actualType, expectedType, key, value);
    }
    indexAndSortDocumentSync(orama, id, indexableProperties, indexableValues, docsCount, language, doc, options);
    if (!skipHooks) {
        (0, hooks_js_1.runSingleHook)(orama.afterInsert, orama, id, doc);
    }
    return id;
}
function validateDocumentProperty(actualType, expectedType, key, value) {
    if ((0, components_js_1.isGeoPointType)(expectedType) &&
        typeof value === 'object' &&
        typeof value.lon === 'number' &&
        typeof value.lat === 'number') {
        return;
    }
    if ((0, components_js_1.isVectorType)(expectedType) && Array.isArray(value))
        return;
    if ((0, components_js_1.isArrayType)(expectedType) && Array.isArray(value))
        return;
    if (ENUM_TYPE.has(expectedType) && STRING_NUMBER_TYPE.has(actualType))
        return;
    if (actualType !== expectedType) {
        throw (0, errors_js_1.createError)('INVALID_DOCUMENT_PROPERTY', key, expectedType, actualType);
    }
}
async function indexAndSortDocument(orama, id, indexableProperties, indexableValues, docsCount, language, doc, options) {
    for (const prop of indexableProperties) {
        const value = indexableValues[prop];
        if (typeof value === 'undefined')
            continue;
        const expectedType = orama.index.getSearchablePropertiesWithTypes(orama.data.index)[prop];
        await orama.index.beforeInsert?.(orama.data.index, prop, id, value, expectedType, language, orama.tokenizer, docsCount);
        const internalId = orama.internalDocumentIDStore.idToInternalId.get(id);
        await orama.index.insert(orama.index, orama.data.index, prop, id, internalId, value, expectedType, language, orama.tokenizer, docsCount, options);
        await orama.index.afterInsert?.(orama.data.index, prop, id, value, expectedType, language, orama.tokenizer, docsCount);
    }
    const sortableProperties = orama.sorter.getSortableProperties(orama.data.sorting);
    const sortableValues = orama.getDocumentProperties(doc, sortableProperties);
    for (const prop of sortableProperties) {
        const value = sortableValues[prop];
        if (typeof value === 'undefined')
            continue;
        const expectedType = orama.sorter.getSortablePropertiesWithTypes(orama.data.sorting)[prop];
        orama.sorter.insert(orama.data.sorting, prop, id, value, expectedType, language);
    }
}
function indexAndSortDocumentSync(orama, id, indexableProperties, indexableValues, docsCount, language, doc, options) {
    for (const prop of indexableProperties) {
        const value = indexableValues[prop];
        if (typeof value === 'undefined')
            continue;
        const expectedType = orama.index.getSearchablePropertiesWithTypes(orama.data.index)[prop];
        const internalDocumentId = (0, internal_document_id_store_js_1.getInternalDocumentId)(orama.internalDocumentIDStore, id);
        orama.index.beforeInsert?.(orama.data.index, prop, id, value, expectedType, language, orama.tokenizer, docsCount);
        orama.index.insert(orama.index, orama.data.index, prop, id, internalDocumentId, value, expectedType, language, orama.tokenizer, docsCount, options);
        orama.index.afterInsert?.(orama.data.index, prop, id, value, expectedType, language, orama.tokenizer, docsCount);
    }
    const sortableProperties = orama.sorter.getSortableProperties(orama.data.sorting);
    const sortableValues = orama.getDocumentProperties(doc, sortableProperties);
    for (const prop of sortableProperties) {
        const value = sortableValues[prop];
        if (typeof value === 'undefined')
            continue;
        const expectedType = orama.sorter.getSortablePropertiesWithTypes(orama.data.sorting)[prop];
        orama.sorter.insert(orama.data.sorting, prop, id, value, expectedType, language);
    }
}
function insertMultiple(orama, docs, batchSize, language, skipHooks, timeout) {
    const asyncNeeded = (0, utils_js_1.isAsyncFunction)(orama.afterInsertMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.beforeInsertMultiple) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.insert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.afterInsert);
    if (asyncNeeded) {
        return innerInsertMultipleAsync(orama, docs, batchSize, language, skipHooks, timeout);
    }
    return innerInsertMultipleSync(orama, docs, batchSize, language, skipHooks, timeout);
}
async function innerInsertMultipleAsync(orama, docs, batchSize = 1000, language, skipHooks, timeout = 0) {
    const ids = [];
    const processNextBatch = async (startIndex) => {
        const endIndex = Math.min(startIndex + batchSize, docs.length);
        const batch = docs.slice(startIndex, endIndex);
        for (const doc of batch) {
            const options = { avlRebalanceThreshold: batch.length };
            const id = await insert(orama, doc, language, skipHooks, options);
            ids.push(id);
        }
        return endIndex;
    };
    const processAllBatches = async () => {
        let currentIndex = 0;
        while (currentIndex < docs.length) {
            const startTime = Date.now();
            currentIndex = await processNextBatch(currentIndex);
            if (timeout > 0) {
                const elapsedTime = Date.now() - startTime;
                const waitTime = timeout - elapsedTime;
                if (waitTime > 0) {
                    (0, utils_js_1.sleep)(waitTime);
                }
            }
        }
    };
    await processAllBatches();
    if (!skipHooks) {
        await (0, hooks_js_1.runMultipleHook)(orama.afterInsertMultiple, orama, docs);
    }
    return ids;
}
function innerInsertMultipleSync(orama, docs, batchSize = 1000, language, skipHooks, timeout = 0) {
    const ids = [];
    let i = 0;
    function processNextBatch() {
        const batch = docs.slice(i * batchSize, (i + 1) * batchSize);
        if (batch.length === 0)
            return false;
        for (const doc of batch) {
            const options = { avlRebalanceThreshold: batch.length };
            const id = insert(orama, doc, language, skipHooks, options);
            ids.push(id);
        }
        i++;
        return true;
    }
    function processAllBatches() {
        const startTime = Date.now();
        // eslint-disable-next-line no-constant-condition
        while (true) {
            const hasMoreBatches = processNextBatch();
            if (!hasMoreBatches)
                break;
            if (timeout > 0) {
                const elapsedTime = Date.now() - startTime;
                if (elapsedTime >= timeout) {
                    const remainingTime = timeout - (elapsedTime % timeout);
                    if (remainingTime > 0) {
                        (0, utils_js_1.sleep)(remainingTime);
                    }
                }
            }
        }
    }
    processAllBatches();
    if (!skipHooks) {
        (0, hooks_js_1.runMultipleHook)(orama.afterInsertMultiple, orama, docs);
    }
    return ids;
}
function innerInsertMultiple(orama, docs, batchSize, language, skipHooks, timeout) {
    const asyncNeeded = (0, utils_js_1.isAsyncFunction)(orama.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.afterInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.beforeInsert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.insert) ||
        (0, utils_js_1.isAsyncFunction)(orama.index.afterInsert);
    if (asyncNeeded) {
        return innerInsertMultipleAsync(orama, docs, batchSize, language, skipHooks, timeout);
    }
    return innerInsertMultipleSync(orama, docs, batchSize, language, skipHooks, timeout);
}
//# sourceMappingURL=insert.js.map