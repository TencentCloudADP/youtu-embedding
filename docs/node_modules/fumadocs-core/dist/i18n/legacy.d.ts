import { createI18nMiddleware as createI18nMiddleware$1 } from './middleware.js';
export { I18nConfig, defineI18n } from './index.js';
import 'next/dist/server/web/types';

/**
 * From what I observed, Next.js will somehow pick "browser" export instead of "import" in middleware.
 * Hence, `createI18nMiddleware` is not available from `fumadocs-core/i18n`, even with compatibility layer.
 *
 * I hope Next.js will fix it in the future, before old projects bump deps and face errors.
 *
 * @deprecated Import from `fumadocs-core/i18n/middleware` instead
 */
declare const createI18nMiddleware: typeof createI18nMiddleware$1;

export { createI18nMiddleware };
