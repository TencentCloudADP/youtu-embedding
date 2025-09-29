import * as react from 'react';
import { AnchorHTMLAttributes } from 'react';

interface LinkProps extends AnchorHTMLAttributes<HTMLAnchorElement> {
    /**
     * If the href is an external URL
     *
     * automatically determined by default
     */
    external?: boolean;
    /**
     * Prefetch links, supported on Next.js
     */
    prefetch?: boolean;
}
declare const Link: react.ForwardRefExoticComponent<LinkProps & react.RefAttributes<HTMLAnchorElement>>;

export { type LinkProps, Link as default };
