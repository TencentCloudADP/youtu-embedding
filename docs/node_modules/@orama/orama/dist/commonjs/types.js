"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.kRemovals = exports.kInsertions = exports.MODE_VECTOR_SEARCH = exports.MODE_HYBRID_SEARCH = exports.MODE_FULLTEXT_SEARCH = void 0;
var constants_js_1 = require("./constants.js");
Object.defineProperty(exports, "MODE_FULLTEXT_SEARCH", { enumerable: true, get: function () { return constants_js_1.MODE_FULLTEXT_SEARCH; } });
Object.defineProperty(exports, "MODE_HYBRID_SEARCH", { enumerable: true, get: function () { return constants_js_1.MODE_HYBRID_SEARCH; } });
Object.defineProperty(exports, "MODE_VECTOR_SEARCH", { enumerable: true, get: function () { return constants_js_1.MODE_VECTOR_SEARCH; } });
exports.kInsertions = Symbol('orama.insertions');
exports.kRemovals = Symbol('orama.removals');
//# sourceMappingURL=types.js.map