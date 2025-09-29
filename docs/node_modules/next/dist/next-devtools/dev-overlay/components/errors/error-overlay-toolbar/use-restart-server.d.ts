export declare function useRestartServer(): {
    restartServer: ({ invalidatePersistentCache, }: {
        invalidatePersistentCache: boolean;
    }) => Promise<void>;
    isPending: boolean;
};
