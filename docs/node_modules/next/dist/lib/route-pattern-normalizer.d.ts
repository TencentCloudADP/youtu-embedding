import type { Token } from 'next/dist/compiled/path-to-regexp';
/**
 * Detects if a route pattern needs normalization for path-to-regexp compatibility.
 */
export declare function hasAdjacentParameterIssues(route: string): boolean;
/**
 * Normalizes route patterns that have adjacent parameters without text between them.
 * Inserts a unique separator that can be safely stripped out later.
 */
export declare function normalizeAdjacentParameters(route: string): string;
/**
 * Normalizes tokens that have repeating modifiers (* or +) but empty prefix and suffix.
 *
 * path-to-regexp 6.3.0+ introduced validation that throws:
 * "Can not repeat without prefix/suffix"
 *
 * This occurs when a token has modifier: '*' or '+' with both prefix: '' and suffix: ''
 */
export declare function normalizeTokensForRegexp(tokens: Token[]): Token[];
/**
 * Strips normalization separators from extracted route parameters.
 * Used by both server and client code to clean up parameters after route matching.
 */
export declare function stripParameterSeparators(params: Record<string, any>): Record<string, any>;
