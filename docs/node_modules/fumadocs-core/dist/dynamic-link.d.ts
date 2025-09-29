import * as react from 'react';
import { LinkProps } from './link.js';

type DynamicLinkProps = LinkProps;
/**
 * Extends the default `Link` component
 *
 * It supports dynamic hrefs, which means you can use `/[lang]/my-page` with `dynamicHrefs` enabled
 */
declare const DynamicLink: react.ForwardRefExoticComponent<LinkProps & react.RefAttributes<HTMLAnchorElement>>;

export { DynamicLink, type DynamicLinkProps, DynamicLink as default };
