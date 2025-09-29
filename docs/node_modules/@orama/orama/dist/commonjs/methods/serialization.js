"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.load = load;
exports.save = save;
function load(orama, raw) {
    orama.internalDocumentIDStore.load(orama, raw.internalDocumentIDStore);
    orama.data.index = orama.index.load(orama.internalDocumentIDStore, raw.index);
    orama.data.docs = orama.documentsStore.load(orama.internalDocumentIDStore, raw.docs);
    orama.data.sorting = orama.sorter.load(orama.internalDocumentIDStore, raw.sorting);
    orama.tokenizer.language = raw.language;
}
function save(orama) {
    return {
        internalDocumentIDStore: orama.internalDocumentIDStore.save(orama.internalDocumentIDStore),
        index: orama.index.save(orama.data.index),
        docs: orama.documentsStore.save(orama.data.docs),
        sorting: orama.sorter.save(orama.data.sorting),
        language: orama.tokenizer.language
    };
}
//# sourceMappingURL=serialization.js.map