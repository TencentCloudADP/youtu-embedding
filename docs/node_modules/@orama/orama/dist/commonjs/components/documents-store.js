"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.create = create;
exports.get = get;
exports.getMultiple = getMultiple;
exports.getAll = getAll;
exports.store = store;
exports.remove = remove;
exports.count = count;
exports.load = load;
exports.save = save;
exports.createDocumentsStore = createDocumentsStore;
const internal_document_id_store_js_1 = require("./internal-document-id-store.js");
function create(_, sharedInternalDocumentStore) {
    return {
        sharedInternalDocumentStore,
        docs: {},
        count: 0
    };
}
function get(store, id) {
    const internalId = (0, internal_document_id_store_js_1.getInternalDocumentId)(store.sharedInternalDocumentStore, id);
    return store.docs[internalId];
}
function getMultiple(store, ids) {
    const idsLength = ids.length;
    const found = Array.from({ length: idsLength });
    for (let i = 0; i < idsLength; i++) {
        const internalId = (0, internal_document_id_store_js_1.getInternalDocumentId)(store.sharedInternalDocumentStore, ids[i]);
        found[i] = store.docs[internalId];
    }
    return found;
}
function getAll(store) {
    return store.docs;
}
function store(store, id, internalId, doc) {
    if (typeof store.docs[internalId] !== 'undefined') {
        return false;
    }
    store.docs[internalId] = doc;
    store.count++;
    return true;
}
function remove(store, id) {
    const internalId = (0, internal_document_id_store_js_1.getInternalDocumentId)(store.sharedInternalDocumentStore, id);
    if (typeof store.docs[internalId] === 'undefined') {
        return false;
    }
    delete store.docs[internalId];
    store.count--;
    return true;
}
function count(store) {
    return store.count;
}
function load(sharedInternalDocumentStore, raw) {
    const rawDocument = raw;
    return {
        docs: rawDocument.docs,
        count: rawDocument.count,
        sharedInternalDocumentStore
    };
}
function save(store) {
    return {
        docs: store.docs,
        count: store.count
    };
}
function createDocumentsStore() {
    return {
        create,
        get,
        getMultiple,
        getAll,
        store,
        remove,
        count,
        load,
        save
    };
}
//# sourceMappingURL=documents-store.js.map