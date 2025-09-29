"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.search = search;
exports.fetchDocumentsWithDistinct = fetchDocumentsWithDistinct;
exports.fetchDocuments = fetchDocuments;
const internal_document_id_store_js_1 = require("../components/internal-document-id-store.js");
const errors_js_1 = require("../errors.js");
const utils_js_1 = require("../utils.js");
const constants_js_1 = require("../constants.js");
const search_fulltext_js_1 = require("./search-fulltext.js");
const search_vector_js_1 = require("./search-vector.js");
const search_hybrid_js_1 = require("./search-hybrid.js");
function search(orama, params, language) {
    const mode = params.mode ?? constants_js_1.MODE_FULLTEXT_SEARCH;
    if (mode === constants_js_1.MODE_FULLTEXT_SEARCH) {
        return (0, search_fulltext_js_1.fullTextSearch)(orama, params, language);
    }
    if (mode === constants_js_1.MODE_VECTOR_SEARCH) {
        return (0, search_vector_js_1.searchVector)(orama, params);
    }
    if (mode === constants_js_1.MODE_HYBRID_SEARCH) {
        return (0, search_hybrid_js_1.hybridSearch)(orama, params);
    }
    throw (0, errors_js_1.createError)('INVALID_SEARCH_MODE', mode);
}
function fetchDocumentsWithDistinct(orama, uniqueDocsArray, offset, limit, distinctOn) {
    const docs = orama.data.docs;
    // Keep track which values we already seen
    const values = new Map();
    // We cannot know how many results we will have in the end,
    // so we need cannot pre-allocate the array.
    const results = [];
    const resultIDs = new Set();
    const uniqueDocsArrayLength = uniqueDocsArray.length;
    let count = 0;
    for (let i = 0; i < uniqueDocsArrayLength; i++) {
        const idAndScore = uniqueDocsArray[i];
        // If there are no more results, just break the loop
        if (typeof idAndScore === 'undefined') {
            continue;
        }
        const [id, score] = idAndScore;
        if (resultIDs.has(id)) {
            continue;
        }
        const doc = orama.documentsStore.get(docs, id);
        const value = (0, utils_js_1.getNested)(doc, distinctOn);
        if (typeof value === 'undefined' || values.has(value)) {
            continue;
        }
        values.set(value, true);
        count++;
        // We shouldn't consider the document if it's not in the offset range
        if (count <= offset) {
            continue;
        }
        results.push({ id: (0, internal_document_id_store_js_1.getDocumentIdFromInternalId)(orama.internalDocumentIDStore, id), score, document: doc });
        resultIDs.add(id);
        // reached the limit, break the loop
        if (count >= offset + limit) {
            break;
        }
    }
    return results;
}
function fetchDocuments(orama, uniqueDocsArray, offset, limit) {
    const docs = orama.data.docs;
    const results = Array.from({
        length: limit
    });
    const resultIDs = new Set();
    // We already have the list of ALL the document IDs containing the search terms.
    // We loop over them starting from a positional value "offset" and ending at "offset + limit"
    // to provide pagination capabilities to the search.
    for (let i = offset; i < limit + offset; i++) {
        const idAndScore = uniqueDocsArray[i];
        // If there are no more results, just break the loop
        if (typeof idAndScore === 'undefined') {
            break;
        }
        const [id, score] = idAndScore;
        if (!resultIDs.has(id)) {
            // We retrieve the full document only AFTER making sure that we really want it.
            // We never retrieve the full document preventively.
            const fullDoc = orama.documentsStore.get(docs, id);
            results[i] = { id: (0, internal_document_id_store_js_1.getDocumentIdFromInternalId)(orama.internalDocumentIDStore, id), score, document: fullDoc };
            resultIDs.add(id);
        }
    }
    return results;
}
//# sourceMappingURL=search.js.map