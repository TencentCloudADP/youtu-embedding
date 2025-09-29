import { BailoutToCSRError } from '../../shared/lib/lazy-dynamic/bailout-to-csr';
import { workAsyncStorage } from '../../server/app-render/work-async-storage.external';
import { workUnitAsyncStorage } from '../../server/app-render/work-unit-async-storage.external';
export function bailoutToClientRendering(reason) {
    const workStore = workAsyncStorage.getStore();
    if (workStore == null ? void 0 : workStore.forceStatic) return;
    const workUnitStore = workUnitAsyncStorage.getStore();
    if (workUnitStore) {
        switch(workUnitStore.type){
            case 'prerender':
            case 'prerender-runtime':
            case 'prerender-client':
            case 'prerender-ppr':
            case 'prerender-legacy':
                throw Object.defineProperty(new BailoutToCSRError(reason), "__NEXT_ERROR_CODE", {
                    value: "E394",
                    enumerable: false,
                    configurable: true
                });
            case 'request':
            case 'cache':
            case 'private-cache':
            case 'unstable-cache':
                break;
            default:
                workUnitStore;
        }
    }
}

//# sourceMappingURL=bailout-to-client-rendering.js.map