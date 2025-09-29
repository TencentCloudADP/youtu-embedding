import {
  defineI18n
} from "../chunk-HUTQC33E.js";
import "../chunk-JSBRDJBE.js";

// src/i18n/legacy.ts
var createI18nMiddleware = (...args) => {
  console.warn(
    "[Fumadocs Core] Please import i18n middleware from `fumadocs-core/i18n/middleware` instead, this export will soon be removed."
  );
  const middleware = import("./middleware.js").then(
    (res) => res.createI18nMiddleware(...args)
  );
  return async (...args2) => (await middleware)(...args2);
};
export {
  createI18nMiddleware,
  defineI18n
};
