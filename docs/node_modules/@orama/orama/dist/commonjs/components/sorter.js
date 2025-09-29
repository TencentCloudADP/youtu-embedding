"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = load;
exports.save = save;
exports.createSorter = createSorter;
const errors_js_1 = require("../errors.js");
const defaults_js_1 = require("./defaults.js");
const internal_document_id_store_js_1 = require("./internal-document-id-store.js");
const utils_js_1 = require("../utils.js");
const languages_js_1 = require("./tokenizer/languages.js");
function innerCreate(orama, sharedInternalDocumentStore, schema, sortableDeniedProperties, prefix) {
    const sorter = {
        language: orama.tokenizer.language,
        sharedInternalDocumentStore,
        enabled: true,
        isSorted: true,
        sortableProperties: [],
        sortablePropertiesWithTypes: {},
        sorts: {}
    };
    for (const [prop, type] of Object.entries(schema)) {
        const path = `${prefix}${prefix ? '.' : ''}${prop}`;
        if (sortableDeniedProperties.includes(path)) {
            continue;
        }
        if (typeof type === 'object' && !Array.isArray(type)) {
            // Nested
            const ret = innerCreate(orama, sharedInternalDocumentStore, type, sortableDeniedProperties, path);
            (0, utils_js_1.safeArrayPush)(sorter.sortableProperties, ret.sortableProperties);
            sorter.sorts = {
                ...sorter.sorts,
                ...ret.sorts
            };
            sorter.sortablePropertiesWithTypes = {
                ...sorter.sortablePropertiesWithTypes,
                ...ret.sortablePropertiesWithTypes
            };
            continue;
        }
        if (!(0, defaults_js_1.isVectorType)(type)) {
            switch (type) {
                case 'boolean':
                case 'number':
                case 'string':
                    sorter.sortableProperties.push(path);
                    sorter.sortablePropertiesWithTypes[path] = type;
                    sorter.sorts[path] = {
                        docs: new Map(),
                        orderedDocsToRemove: new Map(),
                        orderedDocs: [],
                        type: type
                    };
                    break;
                case 'geopoint':
                case 'enum':
                    // We don't allow to sort by enums or geopoints
                    continue;
                case 'enum[]':
                case 'boolean[]':
                case 'number[]':
                case 'string[]':
                    // We don't allow to sort by arrays
                    continue;
                default:
                    throw (0, errors_js_1.createError)('INVALID_SORT_SCHEMA_TYPE', Array.isArray(type) ? 'array' : type, path);
            }
        }
    }
    return sorter;
}
function create(orama, sharedInternalDocumentStore, schema, config) {
    const isSortEnabled = config?.enabled !== false;
    if (!isSortEnabled) {
        return {
            disabled: true
        };
    }
    return innerCreate(orama, sharedInternalDocumentStore, schema, (config || {}).unsortableProperties || [], '');
}
function insert(sorter, prop, id, value) {
    if (!sorter.enabled) {
        return;
    }
    sorter.isSorted = false;
    const internalId = (0, internal_document_id_store_js_1.getInternalDocumentId)(sorter.sharedInternalDocumentStore, id);
    const s = sorter.sorts[prop];
    // This happen during a document updating
    // Because we re-use the same internalId
    // We need to clean-up the data structure before re-inserting
    // to avoid duplicates in the orderedDocs array
    // See: https://github.com/oramasearch/orama/issues/629
    if (s.orderedDocsToRemove.has(internalId)) {
        ensureOrderedDocsAreDeletedByProperty(sorter, prop);
    }
    s.docs.set(internalId, s.orderedDocs.length);
    s.orderedDocs.push([internalId, value]);
}
function ensureIsSorted(sorter) {
    if (sorter.isSorted || !sorter.enabled) {
        return;
    }
    const properties = Object.keys(sorter.sorts);
    for (const prop of properties) {
        ensurePropertyIsSorted(sorter, prop);
    }
    sorter.isSorted = true;
}
function stringSort(language, value, d) {
    return value[1].localeCompare(d[1], (0, languages_js_1.getLocale)(language));
}
function numberSort(value, d) {
    return value[1] - d[1];
}
function booleanSort(value, d) {
    return d[1] ? -1 : 1;
}
function ensurePropertyIsSorted(sorter, prop) {
    const s = sorter.sorts[prop];
    let predicate;
    switch (s.type) {
        case 'string':
            predicate = stringSort.bind(null, sorter.language);
            break;
        case 'number':
            predicate = numberSort.bind(null);
            break;
        case 'boolean':
            predicate = booleanSort.bind(null);
            break;
    }
    s.orderedDocs.sort(predicate);
    // Increment position for the greather documents
    const orderedDocsLength = s.orderedDocs.length;
    for (let i = 0; i < orderedDocsLength; i++) {
        const docId = s.orderedDocs[i][0];
        s.docs.set(docId, i);
    }
}
function ensureOrderedDocsAreDeleted(sorter) {
    const properties = Object.keys(sorter.sorts);
    for (const prop of properties) {
        ensureOrderedDocsAreDeletedByProperty(sorter, prop);
    }
}
function ensureOrderedDocsAreDeletedByProperty(sorter, prop) {
    const s = sorter.sorts[prop];
    if (!s.orderedDocsToRemove.size)
        return;
    s.orderedDocs = s.orderedDocs.filter((doc) => !s.orderedDocsToRemove.has(doc[0]));
    s.orderedDocsToRemove.clear();
}
function remove(sorter, prop, id) {
    if (!sorter.enabled) {
        return;
    }
    const s = sorter.sorts[prop];
    const internalId = (0, internal_document_id_store_js_1.getInternalDocumentId)(sorter.sharedInternalDocumentStore, id);
    const index = s.docs.get(internalId);
    if (!index)
        return;
    s.docs.delete(internalId);
    s.orderedDocsToRemove.set(internalId, true);
}
function sortBy(sorter, docIds, by) {
    if (!sorter.enabled) {
        throw (0, errors_js_1.createError)('SORT_DISABLED');
    }
    const property = by.property;
    const isDesc = by.order === 'DESC';
    const s = sorter.sorts[property];
    if (!s) {
        throw (0, errors_js_1.createError)('UNABLE_TO_SORT_ON_UNKNOWN_FIELD', property, sorter.sortableProperties.join(', '));
    }
    ensureOrderedDocsAreDeletedByProperty(sorter, property);
    ensureIsSorted(sorter);
    docIds.sort((a, b) => {
        // This sort algorithm works leveraging on
        // that s.docs is a map of docId -> position
        // If a document is not indexed, it will be not present in the map
        const indexOfA = s.docs.get((0, internal_document_id_store_js_1.getInternalDocumentId)(sorter.sharedInternalDocumentStore, a[0]));
        const indexOfB = s.docs.get((0, internal_document_id_store_js_1.getInternalDocumentId)(sorter.sharedInternalDocumentStore, b[0]));
        const isAIndexed = typeof indexOfA !== 'undefined';
        const isBIndexed = typeof indexOfB !== 'undefined';
        if (!isAIndexed && !isBIndexed) {
            return 0;
        }
        // unindexed documents are always at the end
        if (!isAIndexed) {
            return 1;
        }
        if (!isBIndexed) {
            return -1;
        }
        return isDesc ? indexOfB - indexOfA : indexOfA - indexOfB;
    });
    return docIds;
}
function getSortableProperties(sorter) {
    if (!sorter.enabled) {
        return [];
    }
    return sorter.sortableProperties;
}
function getSortablePropertiesWithTypes(sorter) {
    if (!sorter.enabled) {
        return {};
    }
    return sorter.sortablePropertiesWithTypes;
}
function load(sharedInternalDocumentStore, raw) {
    const rawDocument = raw;
    if (!rawDocument.enabled) {
        return {
            enabled: false
        };
    }
    const sorts = Object.keys(rawDocument.sorts).reduce((acc, prop) => {
        const { docs, orderedDocs, type } = rawDocument.sorts[prop];
        acc[prop] = {
            docs: new Map(Object.entries(docs).map(([k, v]) => [+k, v])),
            orderedDocsToRemove: new Map(),
            orderedDocs,
            type
        };
        return acc;
    }, {});
    return {
        sharedInternalDocumentStore,
        language: rawDocument.language,
        sortableProperties: rawDocument.sortableProperties,
        sortablePropertiesWithTypes: rawDocument.sortablePropertiesWithTypes,
        sorts,
        enabled: true,
        isSorted: rawDocument.isSorted
    };
}
function save(sorter) {
    if (!sorter.enabled) {
        return {
            enabled: false
        };
    }
    ensureOrderedDocsAreDeleted(sorter);
    ensureIsSorted(sorter);
    const sorts = Object.keys(sorter.sorts).reduce((acc, prop) => {
        const { docs, orderedDocs, type } = sorter.sorts[prop];
        acc[prop] = {
            docs: Object.fromEntries(docs.entries()),
            orderedDocs,
            type
        };
        return acc;
    }, {});
    return {
        language: sorter.language,
        sortableProperties: sorter.sortableProperties,
        sortablePropertiesWithTypes: sorter.sortablePropertiesWithTypes,
        sorts,
        enabled: sorter.enabled,
        isSorted: sorter.isSorted
    };
}
function createSorter() {
    return {
        create,
        insert,
        remove,
        save,
        load,
        sortBy,
        getSortableProperties,
        getSortablePropertiesWithTypes
    };
}
//# sourceMappingURL=sorter.js.map