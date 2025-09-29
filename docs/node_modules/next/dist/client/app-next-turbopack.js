"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
const _appbootstrap = require("./app-bootstrap");
const _onrecoverableerror = require("./react-client-callbacks/on-recoverable-error");
window.next.turbopack = true;
self.__webpack_hash__ = '';
// eslint-disable-next-line @next/internal/typechecked-require
const instrumentationHooks = require('../lib/require-instrumentation-client');
(0, _appbootstrap.appBootstrap)(()=>{
    const { hydrate } = require('./app-index');
    try {
        hydrate(instrumentationHooks);
    } finally{
        if (process.env.NODE_ENV !== 'production') {
            const { getOwnerStack } = require('../next-devtools/userspace/app/errors/stitched-error');
            const { renderAppDevOverlay } = require('next/dist/compiled/next-devtools');
            renderAppDevOverlay(getOwnerStack, _onrecoverableerror.isRecoverableError);
        }
    }
});

if ((typeof exports.default === 'function' || (typeof exports.default === 'object' && exports.default !== null)) && typeof exports.default.__esModule === 'undefined') {
  Object.defineProperty(exports.default, '__esModule', { value: true });
  Object.assign(exports.default, exports);
  module.exports = exports.default;
}

//# sourceMappingURL=app-next-turbopack.js.map