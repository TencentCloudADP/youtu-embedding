const nextInvalidImportErrorLoader = function() {
    const { message } = this.getOptions();
    throw Object.defineProperty(new Error(message), "__NEXT_ERROR_CODE", {
        value: "E394",
        enumerable: false,
        configurable: true
    });
};
export default nextInvalidImportErrorLoader;

//# sourceMappingURL=next-invalid-import-error-loader.js.map