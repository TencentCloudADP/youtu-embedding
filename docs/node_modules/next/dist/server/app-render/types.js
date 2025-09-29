"use strict";
Object.defineProperty(exports, "__esModule", {
    value: true
});
0 && (module.exports = {
    HasLoadingBoundary: null,
    flightRouterStateSchema: null
});
function _export(target, all) {
    for(var name in all)Object.defineProperty(target, name, {
        enumerable: true,
        get: all[name]
    });
}
_export(exports, {
    HasLoadingBoundary: function() {
        return HasLoadingBoundary;
    },
    flightRouterStateSchema: function() {
        return flightRouterStateSchema;
    }
});
const _superstruct = /*#__PURE__*/ _interop_require_default(require("next/dist/compiled/superstruct"));
function _interop_require_default(obj) {
    return obj && obj.__esModule ? obj : {
        default: obj
    };
}
const dynamicParamTypesSchema = _superstruct.default.enums([
    'c',
    'ci',
    'oc',
    'd',
    'di'
]);
const segmentSchema = _superstruct.default.union([
    _superstruct.default.string(),
    _superstruct.default.tuple([
        // Param name
        _superstruct.default.string(),
        // Param cache key (almost the same as the value, but arrays are
        // concatenated into strings)
        // TODO: We should change this to just be the value. Currently we convert
        // it back to a value when passing to useParams. It only needs to be
        // a string when converted to a a cache key, but that doesn't mean we
        // need to store it as that representation.
        _superstruct.default.string(),
        // Dynamic param type
        dynamicParamTypesSchema
    ])
]);
const flightRouterStateSchema = _superstruct.default.tuple([
    segmentSchema,
    _superstruct.default.record(_superstruct.default.string(), _superstruct.default.lazy(()=>flightRouterStateSchema)),
    _superstruct.default.optional(_superstruct.default.nullable(_superstruct.default.string())),
    _superstruct.default.optional(_superstruct.default.nullable(_superstruct.default.union([
        _superstruct.default.literal('refetch'),
        _superstruct.default.literal('refresh'),
        _superstruct.default.literal('inside-shared-layout'),
        _superstruct.default.literal('metadata-only')
    ]))),
    _superstruct.default.optional(_superstruct.default.boolean())
]);
var HasLoadingBoundary = /*#__PURE__*/ function(HasLoadingBoundary) {
    // There is a loading boundary in this particular segment
    HasLoadingBoundary[HasLoadingBoundary["SegmentHasLoadingBoundary"] = 1] = "SegmentHasLoadingBoundary";
    // There is a loading boundary somewhere in the subtree (but not in
    // this segment)
    HasLoadingBoundary[HasLoadingBoundary["SubtreeHasLoadingBoundary"] = 2] = "SubtreeHasLoadingBoundary";
    // There is no loading boundary in this segment or any of its descendants
    HasLoadingBoundary[HasLoadingBoundary["SubtreeHasNoLoadingBoundary"] = 3] = "SubtreeHasNoLoadingBoundary";
    return HasLoadingBoundary;
}({});

//# sourceMappingURL=types.js.map