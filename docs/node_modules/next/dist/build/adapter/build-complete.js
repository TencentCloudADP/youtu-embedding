"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
Object.defineProperty(exports, "handleBuildComplete", {
    enumerable: true,
    get: function() {
        return handleBuildComplete;
    }
});
const _path = /*#__PURE__*/ _interop_require_default(require("path"));
const _promises = /*#__PURE__*/ _interop_require_default(require("fs/promises"));
const _util = require("util");
const _url = require("url");
const _log = /*#__PURE__*/ _interop_require_wildcard(require("../output/log"));
const _glob = /*#__PURE__*/ _interop_require_default(require("next/dist/compiled/glob"));
const _interopdefault = require("../../lib/interop-default");
const _utils = require("../utils");
const _normalizepagepath = require("../../shared/lib/page-path/normalize-page-path");
const _apppaths = require("../../shared/lib/router/utils/app-paths");
const _constants = require("../../shared/lib/constants");
const _renderingmode = require("../rendering-mode");
const _utils1 = require("../../shared/lib/router/utils");
function _interop_require_default(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
function _getRequireWildcardCache(nodeInterop) {
    if (typeof WeakMap !== "function") return null;
    var cacheBabelInterop = new WeakMap();
    var cacheNodeInterop = new WeakMap();
    return (_getRequireWildcardCache = function(nodeInterop) {
        return nodeInterop ? cacheNodeInterop : cacheBabelInterop;
    })(nodeInterop);
}
function _interop_require_wildcard(obj, nodeInterop) {
    if (!nodeInterop && obj && obj.__esModule) {
        return obj;
    }
    if (obj === null || typeof obj !== "object" && typeof obj !== "function") {
        return {
            default: obj
        };
    }
    var cache = _getRequireWildcardCache(nodeInterop);
    if (cache && cache.has(obj)) {
        return cache.get(obj);
    }
    var newObj = {
        __proto__: null
    };
    var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor;
    for(var key in obj){
        if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) {
            var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null;
            if (desc && (desc.get || desc.set)) {
                Object.defineProperty(newObj, key, desc);
            } else {
                newObj[key] = obj[key];
            }
        }
    }
    newObj.default = obj;
    if (cache) {
        cache.set(obj, newObj);
    }
    return newObj;
}
const glob = (0, _util.promisify)(_glob.default);
async function handleBuildComplete({ // dir,
distDir, tracingRoot, adapterPath, pageKeys, appPageKeys, hasNodeMiddleware, hasInstrumentationHook, requiredServerFiles, routesManifest, prerenderManifest, middlewareManifest, functionsConfigManifest, hasStatic404 }) {
    const adapterMod = (0, _interopdefault.interopDefault)(await import((0, _url.pathToFileURL)(require.resolve(adapterPath)).href));
    if (typeof adapterMod.onBuildComplete === 'function') {
        _log.info(`Running onBuildComplete from ${adapterMod.name}`);
        try {
            const outputs = [];
            const staticFiles = await glob('**/*', {
                cwd: _path.default.join(distDir, 'static')
            });
            for (const file of staticFiles){
                const pathname = _path.default.posix.join('/_next/static', file);
                const filePath = _path.default.join(distDir, 'static', file);
                outputs.push({
                    type: _constants.AdapterOutputType.STATIC_FILE,
                    id: _path.default.join('static', file),
                    pathname,
                    filePath
                });
            }
            const sharedNodeAssets = {};
            for (const file of requiredServerFiles){
                // add to shared node assets
                const filePath = _path.default.join(distDir, file);
                const fileOutputPath = _path.default.relative(tracingRoot, filePath);
                sharedNodeAssets[fileOutputPath] = filePath;
            }
            if (hasInstrumentationHook) {
                const assets = await handleTraceFiles(_path.default.join(distDir, 'server', 'instrumentation.js.nft.json'));
                const fileOutputPath = _path.default.relative(tracingRoot, _path.default.join(distDir, 'server', 'instrumentation.js'));
                sharedNodeAssets[fileOutputPath] = _path.default.join(distDir, 'server', 'instrumentation.js');
                Object.assign(sharedNodeAssets, assets);
            }
            async function handleTraceFiles(traceFilePath) {
                const assets = Object.assign({}, sharedNodeAssets);
                const traceData = JSON.parse(await _promises.default.readFile(traceFilePath, 'utf8'));
                const traceFileDir = _path.default.dirname(traceFilePath);
                for (const relativeFile of traceData.files){
                    const tracedFilePath = _path.default.join(traceFileDir, relativeFile);
                    const fileOutputPath = _path.default.relative(tracingRoot, tracedFilePath);
                    assets[fileOutputPath] = tracedFilePath;
                }
                return assets;
            }
            async function handleEdgeFunction(page, isMiddleware = false) {
                let type = _constants.AdapterOutputType.PAGES;
                const isAppPrefix = page.page.startsWith('app/');
                const isAppPage = isAppPrefix && page.page.endsWith('/page');
                const isAppRoute = isAppPrefix && page.page.endsWith('/route');
                if (isMiddleware) {
                    type = _constants.AdapterOutputType.MIDDLEWARE;
                } else if (isAppPage) {
                    type = _constants.AdapterOutputType.APP_PAGE;
                } else if (isAppRoute) {
                    type = _constants.AdapterOutputType.APP_ROUTE;
                } else if (page.page.startsWith('/api')) {
                    type = _constants.AdapterOutputType.PAGES_API;
                }
                const output = {
                    id: page.name,
                    runtime: 'edge',
                    pathname: isAppPrefix ? (0, _apppaths.normalizeAppPath)(page.name) : page.name,
                    filePath: _path.default.join(distDir, 'server', page.files.find((item)=>item.startsWith('server/app') || item.startsWith('server/pages')) || ''),
                    assets: {},
                    type,
                    config: type === _constants.AdapterOutputType.MIDDLEWARE ? {
                        matchers: page.matchers
                    } : {}
                };
                function handleFile(file) {
                    const originalPath = _path.default.join(distDir, file);
                    const fileOutputPath = _path.default.join(_path.default.relative(tracingRoot, distDir), file);
                    if (!output.assets) {
                        output.assets = {};
                    }
                    output.assets[fileOutputPath] = originalPath;
                }
                for (const file of page.files){
                    handleFile(file);
                }
                for (const item of [
                    ...page.wasm || [],
                    ...page.assets || []
                ]){
                    handleFile(item.filePath);
                }
                outputs.push(output);
            }
            const edgeFunctionHandlers = [];
            for (const middleware of Object.values(middlewareManifest.middleware)){
                if ((0, _utils.isMiddlewareFilename)(middleware.name)) {
                    edgeFunctionHandlers.push(handleEdgeFunction(middleware, true));
                }
            }
            for (const page of Object.values(middlewareManifest.functions)){
                edgeFunctionHandlers.push(handleEdgeFunction(page));
            }
            const pagesDistDir = _path.default.join(distDir, 'server', 'pages');
            const pageOutputMap = {};
            for (const page of pageKeys){
                if (middlewareManifest.functions.hasOwnProperty(page)) {
                    continue;
                }
                const route = (0, _normalizepagepath.normalizePagePath)(page);
                const pageFile = _path.default.join(pagesDistDir, `${(0, _normalizepagepath.normalizePagePath)(page)}.js`);
                const pageTraceFile = `${pageFile}.nft.json`;
                const assets = await handleTraceFiles(pageTraceFile).catch((err)=>{
                    if (err.code !== 'ENOENT' || page !== '/404' && page !== '/500') {
                        _log.warn(`Failed to locate traced assets for ${pageFile}`, err);
                    }
                    return {};
                });
                const functionConfig = functionsConfigManifest.functions[route] || {};
                const output = {
                    id: route,
                    type: page.startsWith('/api') ? _constants.AdapterOutputType.PAGES_API : _constants.AdapterOutputType.PAGES,
                    filePath: pageTraceFile.replace(/\.nft\.json$/, ''),
                    pathname: route,
                    assets,
                    runtime: 'nodejs',
                    config: {
                        maxDuration: functionConfig.maxDuration,
                        preferredRegion: functionConfig.regions
                    }
                };
                pageOutputMap[page] = output;
                outputs.push(output);
            }
            if (hasNodeMiddleware) {
                const middlewareFile = _path.default.join(distDir, 'server', 'middleware.js');
                const middlewareTrace = `${middlewareFile}.nft.json`;
                const assets = await handleTraceFiles(middlewareTrace);
                const functionConfig = functionsConfigManifest.functions['/_middleware'] || {};
                outputs.push({
                    pathname: '/_middleware',
                    id: '/_middleware',
                    assets,
                    type: _constants.AdapterOutputType.MIDDLEWARE,
                    runtime: 'nodejs',
                    filePath: middlewareFile,
                    config: {
                        matchers: functionConfig.matchers
                    }
                });
            }
            const appOutputMap = {};
            const appDistDir = _path.default.join(distDir, 'server', 'app');
            if (appPageKeys) {
                for (const page of appPageKeys){
                    if (middlewareManifest.functions.hasOwnProperty(page)) {
                        continue;
                    }
                    const normalizedPage = (0, _apppaths.normalizeAppPath)(page);
                    const pageFile = _path.default.join(appDistDir, `${page}.js`);
                    const pageTraceFile = `${pageFile}.nft.json`;
                    const assets = await handleTraceFiles(pageTraceFile).catch((err)=>{
                        _log.warn(`Failed to copy traced files for ${pageFile}`, err);
                        return {};
                    });
                    const functionConfig = functionsConfigManifest.functions[normalizedPage] || {};
                    const output = {
                        pathname: normalizedPage,
                        id: normalizedPage,
                        assets,
                        type: page.endsWith('/route') ? _constants.AdapterOutputType.APP_ROUTE : _constants.AdapterOutputType.APP_PAGE,
                        runtime: 'nodejs',
                        filePath: pageFile,
                        config: {
                            maxDuration: functionConfig.maxDuration,
                            preferredRegion: functionConfig.regions
                        }
                    };
                    appOutputMap[normalizedPage] = output;
                    outputs.push(output);
                }
            }
            const getParentOutput = (srcRoute, childRoute, allowMissing)=>{
                const parentOutput = pageOutputMap[srcRoute] || appOutputMap[srcRoute];
                if (!parentOutput && !allowMissing) {
                    console.error({
                        appOutputs: Object.keys(appOutputMap),
                        pageOutputs: Object.keys(pageOutputMap)
                    });
                    throw Object.defineProperty(new Error(`Invariant: failed to find source route ${srcRoute} for prerender ${childRoute}`), "__NEXT_ERROR_CODE", {
                        value: "E777",
                        enumerable: false,
                        configurable: true
                    });
                }
                return parentOutput;
            };
            const { prefetchSegmentDirSuffix, prefetchSegmentSuffix, varyHeader, didPostponeHeader, contentTypeHeader } = routesManifest.rsc;
            const handleAppMeta = async (route, initialOutput)=>{
                const meta = JSON.parse(await _promises.default.readFile(_path.default.join(appDistDir, `${route}.meta`), 'utf8').catch(()=>'{}'));
                if (meta.postponed && initialOutput.config) {
                    initialOutput.config.postponed = meta.postponed;
                }
                if (meta == null ? void 0 : meta.segmentPaths) {
                    const segmentsDir = _path.default.join(appDistDir, `${route}${prefetchSegmentDirSuffix}`);
                    for (const segmentPath of meta.segmentPaths){
                        var _initialOutput_fallback, _initialOutput_fallback1, _initialOutput_fallback2;
                        const outputSegmentPath = _path.default.join(appDistDir, route + prefetchSegmentDirSuffix, segmentPath) + prefetchSegmentSuffix;
                        const fallbackPathname = _path.default.join(segmentsDir, segmentPath + prefetchSegmentSuffix);
                        outputs.push({
                            id: outputSegmentPath,
                            pathname: outputSegmentPath,
                            type: _constants.AdapterOutputType.PRERENDER,
                            parentOutputId: initialOutput.parentOutputId,
                            config: {
                                ...initialOutput.config
                            },
                            fallback: {
                                filePath: fallbackPathname,
                                initialExpiration: (_initialOutput_fallback = initialOutput.fallback) == null ? void 0 : _initialOutput_fallback.initialExpiration,
                                initialRevalidate: (_initialOutput_fallback1 = initialOutput.fallback) == null ? void 0 : _initialOutput_fallback1.initialRevalidate,
                                initialHeaders: {
                                    ...(_initialOutput_fallback2 = initialOutput.fallback) == null ? void 0 : _initialOutput_fallback2.initialHeaders,
                                    vary: varyHeader,
                                    'content-type': contentTypeHeader,
                                    [didPostponeHeader]: '2'
                                }
                            }
                        });
                    }
                }
            };
            for(const route in prerenderManifest.routes){
                var _routesManifest_dynamicRoutes_find;
                const { initialExpireSeconds: initialExpiration, initialRevalidateSeconds: initialRevalidate, initialHeaders, initialStatus, prefetchDataRoute, dataRoute, renderingMode, allowHeader, experimentalBypassFor } = prerenderManifest.routes[route];
                const srcRoute = prerenderManifest.routes[route].srcRoute || route;
                const isAppPage = Boolean(appOutputMap[srcRoute]) || srcRoute === '/_not-found';
                const isNotFoundTrue = prerenderManifest.notFoundRoutes.includes(route);
                let allowQuery;
                const routeKeys = (_routesManifest_dynamicRoutes_find = routesManifest.dynamicRoutes.find((item)=>item.page === srcRoute)) == null ? void 0 : _routesManifest_dynamicRoutes_find.routeKeys;
                if (!(0, _utils1.isDynamicRoute)(srcRoute)) {
                    // for non-dynamic routes we use an empty array since
                    // no query values bust the cache for non-dynamic prerenders
                    // prerendered paths also do not pass allowQuery as they match
                    // during handle: 'filesystem' so should not cache differently
                    // by query values
                    allowQuery = [];
                } else if (routeKeys) {
                    // if we have routeKeys in the routes-manifest we use those
                    // for allowQuery for dynamic routes
                    allowQuery = Object.values(routeKeys);
                }
                let filePath = _path.default.join(isAppPage ? appDistDir : pagesDistDir, `${route}.${isAppPage && !dataRoute ? 'body' : 'html'}`);
                // we use the static 404 for notFound: true if available
                // if not we do a blocking invoke on first request
                if (isNotFoundTrue && hasStatic404) {
                    filePath = _path.default.join(pagesDistDir, '404.html');
                }
                const initialOutput = {
                    id: route,
                    type: _constants.AdapterOutputType.PRERENDER,
                    pathname: route,
                    parentOutputId: srcRoute === '/_not-found' ? srcRoute : getParentOutput(srcRoute, route).id,
                    fallback: !isNotFoundTrue || isNotFoundTrue && hasStatic404 ? {
                        filePath,
                        initialStatus,
                        initialHeaders: {
                            ...initialHeaders,
                            vary: varyHeader,
                            'content-type': contentTypeHeader
                        },
                        initialExpiration,
                        initialRevalidate: initialRevalidate || 1
                    } : undefined,
                    config: {
                        allowQuery,
                        allowHeader,
                        renderingMode,
                        bypassFor: experimentalBypassFor,
                        bypassToken: prerenderManifest.preview.previewModeId
                    }
                };
                outputs.push(initialOutput);
                if (dataRoute) {
                    let dataFilePath = _path.default.join(pagesDistDir, `${route}.json`);
                    if (isAppPage) {
                        // When experimental PPR is enabled, we expect that the data
                        // that should be served as a part of the prerender should
                        // be from the prefetch data route. If this isn't enabled
                        // for ppr, the only way to get the data is from the data
                        // route.
                        dataFilePath = _path.default.join(appDistDir, prefetchDataRoute && renderingMode === _renderingmode.RenderingMode.PARTIALLY_STATIC ? prefetchDataRoute : dataRoute);
                    }
                    outputs.push({
                        ...initialOutput,
                        id: dataRoute,
                        pathname: dataRoute,
                        fallback: isNotFoundTrue ? undefined : {
                            ...initialOutput.fallback,
                            filePath: dataFilePath
                        }
                    });
                }
                if (isAppPage) {
                    await handleAppMeta(route, initialOutput);
                }
            }
            for(const dynamicRoute in prerenderManifest.dynamicRoutes){
                var _routesManifest_dynamicRoutes_find1;
                const { fallback, fallbackExpire, fallbackRevalidate, fallbackHeaders, fallbackStatus, allowHeader, dataRoute, renderingMode, experimentalBypassFor } = prerenderManifest.dynamicRoutes[dynamicRoute];
                const isAppPage = Boolean(appOutputMap[dynamicRoute]);
                const allowQuery = Object.values(((_routesManifest_dynamicRoutes_find1 = routesManifest.dynamicRoutes.find((item)=>item.page === dynamicRoute)) == null ? void 0 : _routesManifest_dynamicRoutes_find1.routeKeys) || {});
                const initialOutput = {
                    id: dynamicRoute,
                    type: _constants.AdapterOutputType.PRERENDER,
                    pathname: dynamicRoute,
                    parentOutputId: getParentOutput(dynamicRoute, dynamicRoute).id,
                    config: {
                        allowQuery,
                        allowHeader,
                        renderingMode,
                        bypassFor: experimentalBypassFor,
                        bypassToken: prerenderManifest.preview.previewModeId
                    },
                    fallback: typeof fallback === 'string' ? {
                        filePath: _path.default.join(isAppPage ? appDistDir : pagesDistDir, fallback),
                        initialStatus: fallbackStatus,
                        initialHeaders: fallbackHeaders,
                        initialExpiration: fallbackExpire,
                        initialRevalidate: fallbackRevalidate || 1
                    } : undefined
                };
                outputs.push(initialOutput);
                if (isAppPage) {
                    await handleAppMeta(dynamicRoute, initialOutput);
                }
                if (dataRoute) {
                    outputs.push({
                        ...initialOutput,
                        id: dataRoute,
                        pathname: dataRoute,
                        fallback: undefined
                    });
                }
            }
            await adapterMod.onBuildComplete({
                routes: {
                    dynamicRoutes: routesManifest.dynamicRoutes,
                    rewrites: routesManifest.rewrites,
                    redirects: routesManifest.redirects,
                    headers: routesManifest.headers
                },
                outputs
            });
        } catch (err) {
            _log.error(`Failed to run onBuildComplete from ${adapterMod.name}`);
            throw err;
        }
    }
}

//# sourceMappingURL=build-complete.js.map