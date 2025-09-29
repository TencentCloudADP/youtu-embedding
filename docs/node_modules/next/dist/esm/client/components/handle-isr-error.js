const workAsyncStorage = typeof window === 'undefined' ? require('../../server/app-render/work-async-storage.external').workAsyncStorage : undefined;
// if we are revalidating we want to re-throw the error so the
// function crashes so we can maintain our previous cache
// instead of caching the error page
export function HandleISRError(param) {
    let { error } = param;
    if (workAsyncStorage) {
        const store = workAsyncStorage.getStore();
        if ((store == null ? void 0 : store.isRevalidate) || (store == null ? void 0 : store.isStaticGeneration)) {
            console.error(error);
            throw error;
        }
    }
    return null;
}

//# sourceMappingURL=handle-isr-error.js.map