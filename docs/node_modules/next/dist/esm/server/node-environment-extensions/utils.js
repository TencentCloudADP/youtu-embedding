import { workAsyncStorage } from '../app-render/work-async-storage.external';
import { workUnitAsyncStorage } from '../app-render/work-unit-async-storage.external';
import { abortOnSynchronousPlatformIOAccess, trackSynchronousPlatformIOAccessInDev } from '../app-render/dynamic-rendering';
import { InvariantError } from '../../shared/lib/invariant-error';
export function io(expression, type) {
    const workUnitStore = workUnitAsyncStorage.getStore();
    const workStore = workAsyncStorage.getStore();
    if (!workUnitStore || !workStore) {
        return;
    }
    switch(workUnitStore.type){
        case 'prerender':
        case 'prerender-runtime':
            {
                const prerenderSignal = workUnitStore.controller.signal;
                if (prerenderSignal.aborted === false) {
                    // If the prerender signal is already aborted we don't need to construct
                    // any stacks because something else actually terminated the prerender.
                    let message;
                    switch(type){
                        case 'time':
                            message = `Route "${workStore.route}" used ${expression} instead of using \`performance\` or without explicitly calling \`await connection()\` beforehand. See more info here: https://nextjs.org/docs/messages/next-prerender-current-time`;
                            break;
                        case 'random':
                            message = `Route "${workStore.route}" used ${expression} outside of \`"use cache"\` and without explicitly calling \`await connection()\` beforehand. See more info here: https://nextjs.org/docs/messages/next-prerender-random`;
                            break;
                        case 'crypto':
                            message = `Route "${workStore.route}" used ${expression} outside of \`"use cache"\` and without explicitly calling \`await connection()\` beforehand. See more info here: https://nextjs.org/docs/messages/next-prerender-crypto`;
                            break;
                        default:
                            throw Object.defineProperty(new InvariantError('Unknown expression type in abortOnSynchronousPlatformIOAccess.'), "__NEXT_ERROR_CODE", {
                                value: "E526",
                                enumerable: false,
                                configurable: true
                            });
                    }
                    abortOnSynchronousPlatformIOAccess(workStore.route, expression, applyOwnerStack(Object.defineProperty(new Error(message), "__NEXT_ERROR_CODE", {
                        value: "E394",
                        enumerable: false,
                        configurable: true
                    }), workUnitStore), workUnitStore);
                }
                break;
            }
        case 'prerender-client':
            {
                const prerenderSignal = workUnitStore.controller.signal;
                if (prerenderSignal.aborted === false) {
                    // If the prerender signal is already aborted we don't need to construct
                    // any stacks because something else actually terminated the prerender.
                    let message;
                    switch(type){
                        case 'time':
                            message = `Route "${workStore.route}" used ${expression} inside a Client Component without a Suspense boundary above it. See more info here: https://nextjs.org/docs/messages/next-prerender-current-time-client`;
                            break;
                        case 'random':
                            message = `Route "${workStore.route}" used ${expression} inside a Client Component without a Suspense boundary above it. See more info here: https://nextjs.org/docs/messages/next-prerender-random-client`;
                            break;
                        case 'crypto':
                            message = `Route "${workStore.route}" used ${expression} inside a Client Component without a Suspense boundary above it. See more info here: https://nextjs.org/docs/messages/next-prerender-crypto-client`;
                            break;
                        default:
                            throw Object.defineProperty(new InvariantError('Unknown expression type in abortOnSynchronousPlatformIOAccess.'), "__NEXT_ERROR_CODE", {
                                value: "E526",
                                enumerable: false,
                                configurable: true
                            });
                    }
                    abortOnSynchronousPlatformIOAccess(workStore.route, expression, applyOwnerStack(Object.defineProperty(new Error(message), "__NEXT_ERROR_CODE", {
                        value: "E394",
                        enumerable: false,
                        configurable: true
                    }), workUnitStore), workUnitStore);
                }
                break;
            }
        case 'request':
            if (workUnitStore.prerenderPhase === true) {
                trackSynchronousPlatformIOAccessInDev(workUnitStore);
            }
            break;
        case 'prerender-ppr':
        case 'prerender-legacy':
        case 'cache':
        case 'private-cache':
        case 'unstable-cache':
            break;
        default:
            workUnitStore;
    }
}
function applyOwnerStack(error, workUnitStore) {
    // TODO: Instead of stitching the stacks here, we should log the original
    // error as-is when it occurs, and let `patchErrorInspect` handle adding the
    // owner stack, instead of logging it deferred in the `LogSafely` component
    // via `throwIfDisallowedDynamic`.
    if (process.env.NODE_ENV !== 'production' && workUnitStore.captureOwnerStack) {
        const ownerStack = workUnitStore.captureOwnerStack();
        if (ownerStack) {
            let stack = ownerStack;
            if (error.stack) {
                const frames = [];
                for (const frame of error.stack.split('\n').slice(1)){
                    if (frame.includes('react_stack_bottom_frame')) {
                        break;
                    }
                    frames.push(frame);
                }
                stack = '\n' + frames.join('\n') + stack;
            }
            error.stack = error.name + ': ' + error.message + stack;
        }
    }
    return error;
}

//# sourceMappingURL=utils.js.map