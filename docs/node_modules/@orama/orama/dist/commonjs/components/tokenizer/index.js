"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.normalizeToken = normalizeToken;
exports.createTokenizer = createTokenizer;
const errors_js_1 = require("../../errors.js");
const diacritics_js_1 = require("./diacritics.js");
const languages_js_1 = require("./languages.js");
const english_stemmer_js_1 = require("./english-stemmer.js");
function normalizeToken(prop, token, withCache = true) {
    const key = `${this.language}:${prop}:${token}`;
    if (withCache && this.normalizationCache.has(key)) {
        return this.normalizationCache.get(key);
    }
    // Remove stopwords if enabled
    if (this.stopWords?.includes(token)) {
        if (withCache) {
            this.normalizationCache.set(key, '');
        }
        return '';
    }
    // Apply stemming if enabled
    if (this.stemmer && !this.stemmerSkipProperties.has(prop)) {
        token = this.stemmer(token);
    }
    token = (0, diacritics_js_1.replaceDiacritics)(token);
    if (withCache) {
        this.normalizationCache.set(key, token);
    }
    return token;
}
/* c8 ignore next 10 */
function trim(text) {
    while (text[text.length - 1] === '') {
        text.pop();
    }
    while (text[0] === '') {
        text.shift();
    }
    return text;
}
function tokenize(input, language, prop, withCache = true) {
    if (language && language !== this.language) {
        throw (0, errors_js_1.createError)('LANGUAGE_NOT_SUPPORTED', language);
    }
    /* c8 ignore next 3 */
    if (typeof input !== 'string') {
        return [input];
    }
    const normalizeToken = this.normalizeToken.bind(this, prop ?? '');
    let tokens;
    if (prop && this.tokenizeSkipProperties.has(prop)) {
        tokens = [normalizeToken(input, withCache)];
    }
    else {
        const splitRule = languages_js_1.SPLITTERS[this.language];
        tokens = input
            .toLowerCase()
            .split(splitRule)
            .map(t => normalizeToken(t, withCache))
            .filter(Boolean);
    }
    const trimTokens = trim(tokens);
    if (!this.allowDuplicates) {
        return Array.from(new Set(trimTokens));
    }
    return trimTokens;
}
function createTokenizer(config = {}) {
    if (!config.language) {
        config.language = 'english';
    }
    else if (!languages_js_1.SUPPORTED_LANGUAGES.includes(config.language)) {
        throw (0, errors_js_1.createError)('LANGUAGE_NOT_SUPPORTED', config.language);
    }
    // Handle stemming - It is disabled by default
    let stemmer;
    if (config.stemming || (config.stemmer && !('stemming' in config))) {
        if (config.stemmer) {
            if (typeof config.stemmer !== 'function') {
                throw (0, errors_js_1.createError)('INVALID_STEMMER_FUNCTION_TYPE');
            }
            stemmer = config.stemmer;
        }
        else {
            if (config.language === 'english') {
                stemmer = english_stemmer_js_1.stemmer;
            }
            else {
                throw (0, errors_js_1.createError)('MISSING_STEMMER', config.language);
            }
        }
    }
    // Handle stopwords
    let stopWords;
    if (config.stopWords !== false) {
        stopWords = [];
        if (Array.isArray(config.stopWords)) {
            stopWords = config.stopWords;
        }
        else if (typeof config.stopWords === 'function') {
            stopWords = config.stopWords(stopWords);
        }
        else if (config.stopWords) {
            throw (0, errors_js_1.createError)('CUSTOM_STOP_WORDS_MUST_BE_FUNCTION_OR_ARRAY');
        }
        // Make sure stopWords is just an array of strings
        if (!Array.isArray(stopWords)) {
            throw (0, errors_js_1.createError)('CUSTOM_STOP_WORDS_MUST_BE_FUNCTION_OR_ARRAY');
        }
        for (const s of stopWords) {
            if (typeof s !== 'string') {
                throw (0, errors_js_1.createError)('CUSTOM_STOP_WORDS_MUST_BE_FUNCTION_OR_ARRAY');
            }
        }
    }
    // Create the tokenizer
    const tokenizer = {
        tokenize,
        language: config.language,
        stemmer,
        stemmerSkipProperties: new Set(config.stemmerSkipProperties ? [config.stemmerSkipProperties].flat() : []),
        tokenizeSkipProperties: new Set(config.tokenizeSkipProperties ? [config.tokenizeSkipProperties].flat() : []),
        stopWords,
        allowDuplicates: Boolean(config.allowDuplicates),
        normalizeToken,
        normalizationCache: new Map()
    };
    tokenizer.tokenize = tokenize.bind(tokenizer);
    tokenizer.normalizeToken = normalizeToken;
    return tokenizer;
}
//# sourceMappingURL=index.js.map