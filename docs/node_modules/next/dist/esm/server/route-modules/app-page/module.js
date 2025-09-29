import { renderToHTMLOrFlight } from '../../app-render/app-render';
import { RouteModule } from '../route-module';
import * as vendoredContexts from './vendored/contexts/entrypoints';
import { PrerenderManifestMatcher } from './helpers/prerender-manifest-matcher';
import { NEXT_ROUTER_PREFETCH_HEADER, NEXT_ROUTER_SEGMENT_PREFETCH_HEADER, NEXT_ROUTER_STATE_TREE_HEADER, NEXT_URL, RSC_HEADER } from '../../../client/components/app-router-headers';
import { isInterceptionRouteAppPath } from '../../../shared/lib/router/utils/interception-routes';
let vendoredReactRSC;
let vendoredReactSSR;
// the vendored Reacts are loaded from their original source in the edge runtime
if (process.env.NEXT_RUNTIME !== 'edge') {
    vendoredReactRSC = require('./vendored/rsc/entrypoints');
    vendoredReactSSR = require('./vendored/ssr/entrypoints');
}
export class AppPageRouteModule extends RouteModule {
    constructor(options){
        super(options), this.matchers = new WeakMap();
        this.isAppRouter = true;
    }
    match(pathname, prerenderManifest) {
        // Lazily create the matcher based on the provided prerender manifest.
        let matcher = this.matchers.get(prerenderManifest);
        if (!matcher) {
            matcher = new PrerenderManifestMatcher(this.definition.pathname, prerenderManifest);
            this.matchers.set(prerenderManifest, matcher);
        }
        // Match the pathname to the dynamic route.
        return matcher.match(pathname);
    }
    render(req, res, context) {
        return renderToHTMLOrFlight(req, res, context.page, context.query, context.fallbackRouteParams, context.renderOpts, context.serverComponentsHmrCache, false, context.sharedContext);
    }
    warmup(req, res, context) {
        return renderToHTMLOrFlight(req, res, context.page, context.query, context.fallbackRouteParams, context.renderOpts, context.serverComponentsHmrCache, true, context.sharedContext);
    }
    pathCouldBeIntercepted(resolvedPathname, interceptionRoutePatterns) {
        return isInterceptionRouteAppPath(resolvedPathname) || interceptionRoutePatterns.some((regexp)=>{
            return regexp.test(resolvedPathname);
        });
    }
    getVaryHeader(resolvedPathname, interceptionRoutePatterns) {
        const baseVaryHeader = `${RSC_HEADER}, ${NEXT_ROUTER_STATE_TREE_HEADER}, ${NEXT_ROUTER_PREFETCH_HEADER}, ${NEXT_ROUTER_SEGMENT_PREFETCH_HEADER}`;
        if (this.pathCouldBeIntercepted(resolvedPathname, interceptionRoutePatterns)) {
            // Interception route responses can vary based on the `Next-URL` header.
            // We use the Vary header to signal this behavior to the client to properly cache the response.
            return `${baseVaryHeader}, ${NEXT_URL}`;
        } else {
            // We don't need to include `Next-URL` in the Vary header for non-interception routes since it won't affect the response.
            // We also set this header for pages to avoid caching issues when navigating between pages and app.
            return baseVaryHeader;
        }
    }
}
const vendored = {
    'react-rsc': vendoredReactRSC,
    'react-ssr': vendoredReactSSR,
    contexts: vendoredContexts
};
export { renderToHTMLOrFlight, vendored };
export default AppPageRouteModule;

//# sourceMappingURL=module.js.map