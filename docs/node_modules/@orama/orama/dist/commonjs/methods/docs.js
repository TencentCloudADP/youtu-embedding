"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getByID = getByID;
exports.count = count;
function getByID(db, id) {
    return db.documentsStore.get(db.data.docs, id);
}
function count(db) {
    return db.documentsStore.count(db.data.docs);
}
//# sourceMappingURL=docs.js.map