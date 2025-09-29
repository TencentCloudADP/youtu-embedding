'use client';
"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
Object.defineProperty(exports, "AsyncMetadataOutlet", {
    enumerable: true,
    get: function() {
        return AsyncMetadataOutlet;
    }
});
const _jsxruntime = require("react/jsx-runtime");
const _react = require("react");
function MetadataOutlet(param) {
    let { promise } = param;
    const { error, digest } = (0, _react.use)(promise);
    if (error) {
        if (digest) {
            // The error will lose its original digest after passing from server layer to client layer；
            // We recover the digest property here to override the React created one if original digest exists.
            ;
            error.digest = digest;
        }
        throw error;
    }
    return null;
}
function AsyncMetadataOutlet(param) {
    let { promise } = param;
    return /*#__PURE__*/ (0, _jsxruntime.jsx)(_react.Suspense, {
        fallback: null,
        children: /*#__PURE__*/ (0, _jsxruntime.jsx)(MetadataOutlet, {
            promise: promise
        })
    });
}

if ((typeof exports.default === 'function' || (typeof exports.default === 'object' && exports.default !== null)) && typeof exports.default.__esModule === 'undefined') {
  Object.defineProperty(exports.default, '__esModule', { value: true });
  Object.assign(exports.default, exports);
  module.exports = exports.default;
}

//# sourceMappingURL=async-metadata.js.map