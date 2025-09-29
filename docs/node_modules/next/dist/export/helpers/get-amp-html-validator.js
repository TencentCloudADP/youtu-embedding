"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
0 && (module.exports = {
    getAmpValidatorInstance: null,
    getBundledAmpValidatorFilepath: null
});
function _export(target, all) {
    for(var name in all)Object.defineProperty(target, name, {
        enumerable: true,
        get: all[name]
    });
}
_export(exports, {
    getAmpValidatorInstance: function() {
        return getAmpValidatorInstance;
    },
    getBundledAmpValidatorFilepath: function() {
        return getBundledAmpValidatorFilepath;
    }
});
const _amphtmlvalidator = /*#__PURE__*/ _interop_require_default(require("next/dist/compiled/amphtml-validator"));
function _interop_require_default(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
const instancePromises = new Map();
function getAmpValidatorInstance(validatorPath) {
    let promise = instancePromises.get(validatorPath);
    if (!promise) {
        // NOTE: if `validatorPath` is undefined, `AmpHtmlValidator` will load the code from its default URL
        promise = _amphtmlvalidator.default.getInstance(validatorPath);
        instancePromises.set(validatorPath, promise);
    }
    return promise;
}
function getBundledAmpValidatorFilepath() {
    return require.resolve('next/dist/compiled/amphtml-validator/validator_wasm.js');
}

//# sourceMappingURL=get-amp-html-validator.js.map