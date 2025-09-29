"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
Object.defineProperty(exports, "default", {
    enumerable: true,
    get: function() {
        return _default;
    }
});
const nextInvalidImportErrorLoader = function() {
    const { message } = this.getOptions();
    throw Object.defineProperty(new Error(message), "__NEXT_ERROR_CODE", {
        value: "E394",
        enumerable: false,
        configurable: true
    });
};
const _default = nextInvalidImportErrorLoader;

//# sourceMappingURL=next-invalid-import-error-loader.js.map