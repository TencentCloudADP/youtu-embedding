export function mergeRefs(...refs) {
    return (value) => {
        refs.forEach((ref) => {
            if (typeof ref === 'function') {
                ref(value);
            }
            else if (ref) {
                ref.current = value;
            }
        });
    };
}
