import { jsx as _jsx } from "react/jsx-runtime";
import defaultMdxComponents from './mdx.js';
import * as path from 'node:path';
/**
 * Extend the default Link component to resolve relative file paths in `href`.
 *
 * @param page the current page
 * @param source the source object
 * @param OverrideLink The component to override from
 */
export function createRelativeLink(source, page, OverrideLink = defaultMdxComponents.a) {
    return async function RelativeLink({ href, ...props }) {
        // resolve relative href
        if (href && href.startsWith('.')) {
            const target = source.getPageByHref(href, {
                dir: path.dirname(page.path),
                language: page.locale,
            });
            if (target) {
                href = target.hash
                    ? `${target.page.url}#${target.hash}`
                    : target.page.url;
            }
        }
        return _jsx(OverrideLink, { href: href, ...props });
    };
}
export { defaultMdxComponents as default };
