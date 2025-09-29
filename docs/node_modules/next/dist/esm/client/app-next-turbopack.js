import { appBootstrap } from './app-bootstrap';
import { isRecoverableError } from './react-client-callbacks/on-recoverable-error';
window.next.turbopack = true;
self.__webpack_hash__ = '';
// eslint-disable-next-line @next/internal/typechecked-require
const instrumentationHooks = require('../lib/require-instrumentation-client');
appBootstrap(()=>{
    const { hydrate } = require('./app-index');
    try {
        hydrate(instrumentationHooks);
    } finally{
        if (process.env.NODE_ENV !== 'production') {
            const { getOwnerStack } = require('../next-devtools/userspace/app/errors/stitched-error');
            const { renderAppDevOverlay } = require('next/dist/compiled/next-devtools');
            renderAppDevOverlay(getOwnerStack, isRecoverableError);
        }
    }
});

//# sourceMappingURL=app-next-turbopack.js.map