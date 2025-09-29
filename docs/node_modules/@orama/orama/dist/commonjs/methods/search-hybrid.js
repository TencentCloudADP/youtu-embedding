"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.innerHybridSearch = innerHybridSearch;
exports.hybridSearch = hybridSearch;
const utils_js_1 = require("../utils.js");
const facets_js_1 = require("../components/facets.js");
const groups_js_1 = require("../components/groups.js");
const search_js_1 = require("./search.js");
const search_fulltext_js_1 = require("./search-fulltext.js");
const search_vector_js_1 = require("./search-vector.js");
const hooks_js_1 = require("../components/hooks.js");
function innerHybridSearch(orama, params, language) {
    const fullTextIDs = minMaxScoreNormalization((0, search_fulltext_js_1.innerFullTextSearch)(orama, params, language));
    const vectorIDs = (0, search_vector_js_1.innerVectorSearch)(orama, params, language);
    const hybridWeights = params.hybridWeights;
    return mergeAndRankResults(fullTextIDs, vectorIDs, params.term ?? '', hybridWeights);
}
function hybridSearch(orama, params, language) {
    const timeStart = (0, utils_js_1.getNanosecondsTime)();
    function performSearchLogic() {
        const uniqueTokenScores = innerHybridSearch(orama, params, language);
        let facetsResults;
        const shouldCalculateFacets = params.facets && Object.keys(params.facets).length > 0;
        if (shouldCalculateFacets) {
            facetsResults = (0, facets_js_1.getFacets)(orama, uniqueTokenScores, params.facets);
        }
        let groups;
        if (params.groupBy) {
            groups = (0, groups_js_1.getGroups)(orama, uniqueTokenScores, params.groupBy);
        }
        const offset = params.offset ?? 0;
        const limit = params.limit ?? 10;
        const results = (0, search_js_1.fetchDocuments)(orama, uniqueTokenScores, offset, limit).filter(Boolean);
        const timeEnd = (0, utils_js_1.getNanosecondsTime)();
        const returningResults = {
            count: uniqueTokenScores.length,
            elapsed: {
                raw: Number(timeEnd - timeStart),
                formatted: (0, utils_js_1.formatNanoseconds)(timeEnd - timeStart)
            },
            hits: results,
            ...(facetsResults ? { facets: facetsResults } : {}),
            ...(groups ? { groups } : {})
        };
        const includeVectors = params.includeVectors ?? false;
        if (!includeVectors) {
            const vectorProperties = Object.keys(orama.data.index.vectorIndexes);
            (0, utils_js_1.removeVectorsFromHits)(returningResults, vectorProperties);
        }
        return returningResults;
    }
    async function executeSearchAsync() {
        if (orama.beforeSearch) {
            await (0, hooks_js_1.runBeforeSearch)(orama.beforeSearch, orama, params, language);
        }
        const results = performSearchLogic();
        if (orama.afterSearch) {
            await (0, hooks_js_1.runAfterSearch)(orama.afterSearch, orama, params, language, results);
        }
        return results;
    }
    const asyncNeeded = orama.beforeSearch?.length || orama.afterSearch?.length;
    if (asyncNeeded) {
        return executeSearchAsync();
    }
    return performSearchLogic();
}
function extractScore(token) {
    return token[1];
}
function minMaxScoreNormalization(results) {
    // In this case I disabled the `prefer-spread` rule because spread seems to be slower
    // eslint-disable-next-line prefer-spread
    const maxScore = Math.max.apply(Math, results.map(extractScore));
    return results.map(([id, score]) => [id, score / maxScore]);
}
function normalizeScore(score, maxScore) {
    return score / maxScore;
}
function hybridScoreBuilder(textWeight, vectorWeight) {
    return (textScore, vectorScore) => textScore * textWeight + vectorScore * vectorWeight;
}
function mergeAndRankResults(textResults, vectorResults, query, hybridWeights) {
    // eslint-disable-next-line prefer-spread
    const maxTextScore = Math.max.apply(Math, textResults.map(extractScore));
    // eslint-disable-next-line prefer-spread
    const maxVectorScore = Math.max.apply(Math, vectorResults.map(extractScore));
    const hasHybridWeights = hybridWeights && hybridWeights.text && hybridWeights.vector;
    const { text: textWeight, vector: vectorWeight } = hasHybridWeights ? hybridWeights : getQueryWeights(query);
    const mergedResults = new Map();
    const textResultsLength = textResults.length;
    const hybridScore = hybridScoreBuilder(textWeight, vectorWeight);
    for (let i = 0; i < textResultsLength; i++) {
        const [id, score] = textResults[i];
        const normalizedScore = normalizeScore(score, maxTextScore);
        const hybridScoreValue = hybridScore(normalizedScore, 0);
        mergedResults.set(id, hybridScoreValue);
    }
    const vectorResultsLength = vectorResults.length;
    for (let i = 0; i < vectorResultsLength; i++) {
        const [resultId, score] = vectorResults[i];
        const normalizedScore = normalizeScore(score, maxVectorScore);
        const existingRes = mergedResults.get(resultId) ?? 0;
        mergedResults.set(resultId, existingRes + hybridScore(0, normalizedScore));
    }
    return [...mergedResults].sort((a, b) => b[1] - a[1]);
}
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function getQueryWeights(query) {
    // In the next versions of Orama, we will ship a plugin containing a ML model to adjust the weights
    // based on whether the query is keyword-focused, conceptual, etc.
    // For now, we just return a fixed value.
    return {
        text: 0.5,
        vector: 0.5
    };
}
//# sourceMappingURL=search-hybrid.js.map