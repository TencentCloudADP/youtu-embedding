import { bundledLanguages, getSingletonHighlighter } from 'shiki';
import rehypeShikiFromHighlighter from './core.mjs';
import 'shiki/core';
import 'unist-util-visit';
import 'hast-util-to-string';

const rehypeShiki = function(options = {}) {
  const themeNames = ("themes" in options ? Object.values(options.themes) : [options.theme]).filter(Boolean);
  const langs = options.langs || Object.keys(bundledLanguages);
  const langAlias = options.langAlias || {};
  let getHandler;
  return async (tree) => {
    if (!getHandler) {
      getHandler = getSingletonHighlighter({
        themes: themeNames,
        langs,
        langAlias
      }).then((highlighter) => rehypeShikiFromHighlighter.call(this, highlighter, options));
    }
    const handler = await getHandler;
    return handler(tree);
  };
};

export { rehypeShiki as default };
