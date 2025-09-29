/**
 * Don't use this, could be deleted anytime.
 *
 * @internal
 */
declare function useEffectEvent<F extends (...params: never[]) => unknown>(callback: F): F;

export { useEffectEvent };
