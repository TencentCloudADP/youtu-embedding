import type { ComponentProps } from 'react';
import { type ButtonProps } from '../../components/ui/button.js';
interface SearchToggleProps extends Omit<ComponentProps<'button'>, 'color'>, ButtonProps {
    hideIfDisabled?: boolean;
}
export declare function SearchToggle({ hideIfDisabled, size, color, ...props }: SearchToggleProps): import("react/jsx-runtime").JSX.Element | null;
export declare function LargeSearchToggle({ hideIfDisabled, ...props }: ComponentProps<'button'> & {
    hideIfDisabled?: boolean;
}): import("react/jsx-runtime").JSX.Element | null;
export {};
//# sourceMappingURL=search-toggle.d.ts.map