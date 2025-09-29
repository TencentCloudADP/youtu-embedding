// src/highlight/shiki.ts
import { toJsxRuntime } from "hast-util-to-jsx-runtime";
import { Fragment } from "react";
import { jsx, jsxs } from "react/jsx-runtime";
var defaultThemes = {
  light: "github-light",
  dark: "github-dark"
};
var highlighters = /* @__PURE__ */ new Map();
async function highlightHast(code, options) {
  const {
    lang: initialLang,
    fallbackLanguage,
    components: _,
    engine = "oniguruma",
    ...rest
  } = options;
  let lang = initialLang;
  let themes;
  let themesToLoad;
  if ("theme" in options && options.theme) {
    themes = { theme: options.theme };
    themesToLoad = [themes.theme];
  } else {
    themes = {
      themes: "themes" in options && options.themes ? options.themes : defaultThemes
    };
    themesToLoad = Object.values(themes.themes).filter((v) => v !== void 0);
  }
  let highlighter;
  if (typeof engine === "string") {
    highlighter = await getHighlighter(engine, {
      langs: [],
      themes: themesToLoad
    });
  } else {
    highlighter = await getHighlighter("custom", {
      engine,
      langs: [],
      themes: themesToLoad
    });
    if (process.env.NODE_ENV === "development") {
      console.warn(
        "[Fumadocs `highlight()`] Avoid passing `engine` directly. For custom engines, use `shiki` directly instead."
      );
    }
  }
  try {
    await highlighter.loadLanguage(lang);
  } catch {
    lang = fallbackLanguage ?? "text";
    await highlighter.loadLanguage(lang);
  }
  return highlighter.codeToHast(code, {
    lang,
    ...rest,
    ...themes,
    defaultColor: "themes" in themes ? false : void 0
  });
}
function _renderHighlight(hast, options) {
  return toJsxRuntime(hast, {
    jsx,
    jsxs,
    development: false,
    components: options?.components,
    Fragment
  });
}
async function getHighlighter(engineType, options) {
  const { createHighlighter } = await import("shiki");
  let highlighter = highlighters.get(engineType);
  if (!highlighter) {
    let engine;
    if (engineType === "js") {
      engine = import("shiki/engine/javascript").then(
        (res) => res.createJavaScriptRegexEngine()
      );
    } else if (engineType === "oniguruma" || !options.engine) {
      engine = import("shiki/engine/oniguruma").then(
        (res) => res.createOnigurumaEngine(import("shiki/wasm"))
      );
    } else {
      engine = options.engine;
    }
    highlighter = createHighlighter({
      ...options,
      engine
    });
    highlighters.set(engineType, highlighter);
    return highlighter;
  }
  return highlighter.then(async (instance) => {
    await Promise.all([
      // @ts-expect-error unknown
      instance.loadLanguage(...options.langs),
      // @ts-expect-error unknown
      instance.loadTheme(...options.themes)
    ]);
    return instance;
  });
}
async function highlight(code, options) {
  return _renderHighlight(await highlightHast(code, options), options);
}

export {
  defaultThemes,
  highlightHast,
  getHighlighter,
  highlight
};
