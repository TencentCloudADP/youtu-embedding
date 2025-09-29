import type { ComponentPropsWithoutRef } from 'react';
type Types = 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6';
type HeadingProps<T extends Types> = Omit<ComponentPropsWithoutRef<T>, 'as'> & {
    as?: T;
};
export declare function Heading<T extends Types = 'h1'>({ as, className, ...props }: HeadingProps<T>): React.ReactElement;
export {};
//# sourceMappingURL=heading.d.ts.map