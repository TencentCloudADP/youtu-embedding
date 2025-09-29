import * as react_jsx_runtime from 'react/jsx-runtime';
import { ReactNode, HTMLAttributes, FC } from 'react';

declare function HideIfEmptyProvider({ nonce, children, }: {
    nonce?: string;
    children: ReactNode;
}): react_jsx_runtime.JSX.Element;
/**
 * The built-in CSS `:empty` selector cannot detect if the children is hidden, classes such as `md:hidden` causes it to fail.
 * This component is an enhancement to `empty:hidden` to fix the issue described above.
 *
 * This can be expensive, please avoid this whenever possible.
 */
declare function HideIfEmpty<Props extends HTMLAttributes<HTMLElement>>({ as: Comp, ...props }: Props & {
    as: FC<Props>;
}): react_jsx_runtime.JSX.Element;

export { HideIfEmpty, HideIfEmptyProvider };
