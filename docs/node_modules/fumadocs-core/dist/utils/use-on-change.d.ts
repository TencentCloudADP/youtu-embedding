/**
 * @param value - state to watch
 * @param onChange - when the state changed
 * @param isUpdated - a function that determines if the state is updated
 */
declare function useOnChange<T>(value: T, onChange: (current: T, previous: T) => void, isUpdated?: (prev: T, current: T) => boolean): void;

export { useOnChange };
