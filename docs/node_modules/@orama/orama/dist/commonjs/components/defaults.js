"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getDocumentProperties = void 0;
exports.formatElapsedTime = formatElapsedTime;
exports.getDocumentIndexId = getDocumentIndexId;
exports.validateSchema = validateSchema;
exports.isGeoPointType = isGeoPointType;
exports.isVectorType = isVectorType;
exports.isArrayType = isArrayType;
exports.getInnerType = getInnerType;
exports.getVectorSize = getVectorSize;
const errors_js_1 = require("../errors.js");
const utils_js_1 = require("../utils.js");
var utils_js_2 = require("../utils.js");
Object.defineProperty(exports, "getDocumentProperties", { enumerable: true, get: function () { return utils_js_2.getDocumentProperties; } });
function formatElapsedTime(n) {
    return {
        raw: Number(n),
        formatted: (0, utils_js_1.formatNanoseconds)(n)
    };
}
function getDocumentIndexId(doc) {
    if (doc.id) {
        if (typeof doc.id !== 'string') {
            throw (0, errors_js_1.createError)('DOCUMENT_ID_MUST_BE_STRING', typeof doc.id);
        }
        return doc.id;
    }
    return (0, utils_js_1.uniqueId)();
}
function validateSchema(doc, schema) {
    for (const [prop, type] of Object.entries(schema)) {
        const value = doc[prop];
        if (typeof value === 'undefined') {
            continue;
        }
        if (type === 'geopoint' &&
            typeof value === 'object' &&
            typeof value.lon === 'number' &&
            typeof value.lat === 'number') {
            continue;
        }
        if (type === 'enum' && (typeof value === 'string' || typeof value === 'number')) {
            continue;
        }
        if (type === 'enum[]' && Array.isArray(value)) {
            const valueLength = value.length;
            for (let i = 0; i < valueLength; i++) {
                if (typeof value[i] !== 'string' && typeof value[i] !== 'number') {
                    return prop + '.' + i;
                }
            }
            continue;
        }
        if (isVectorType(type)) {
            const vectorSize = getVectorSize(type);
            if (!Array.isArray(value) || value.length !== vectorSize) {
                throw (0, errors_js_1.createError)('INVALID_INPUT_VECTOR', prop, vectorSize, value.length);
            }
            continue;
        }
        if (isArrayType(type)) {
            if (!Array.isArray(value)) {
                return prop;
            }
            const expectedType = getInnerType(type);
            const valueLength = value.length;
            for (let i = 0; i < valueLength; i++) {
                if (typeof value[i] !== expectedType) {
                    return prop + '.' + i;
                }
            }
            continue;
        }
        if (typeof type === 'object') {
            if (!value || typeof value !== 'object') {
                return prop;
            }
            // using as ResultDocument is not exactly right but trying to be type-safe here is not useful
            const subProp = validateSchema(value, type);
            if (subProp) {
                return prop + '.' + subProp;
            }
            continue;
        }
        if (typeof value !== type) {
            return prop;
        }
    }
    return undefined;
}
const IS_ARRAY_TYPE = {
    string: false,
    number: false,
    boolean: false,
    enum: false,
    geopoint: false,
    'string[]': true,
    'number[]': true,
    'boolean[]': true,
    'enum[]': true
};
const INNER_TYPE = {
    'string[]': 'string',
    'number[]': 'number',
    'boolean[]': 'boolean',
    'enum[]': 'enum'
};
function isGeoPointType(type) {
    return type === 'geopoint';
}
function isVectorType(type) {
    return typeof type === 'string' && /^vector\[\d+\]$/.test(type);
}
function isArrayType(type) {
    return typeof type === 'string' && IS_ARRAY_TYPE[type];
}
function getInnerType(type) {
    return INNER_TYPE[type];
}
function getVectorSize(type) {
    const size = Number(type.slice(7, -1));
    switch (true) {
        case isNaN(size):
            throw (0, errors_js_1.createError)('INVALID_VECTOR_VALUE', type);
        case size <= 0:
            throw (0, errors_js_1.createError)('INVALID_VECTOR_SIZE', type);
        default:
            return size;
    }
}
//# sourceMappingURL=defaults.js.map