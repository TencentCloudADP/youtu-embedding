import "../chunk-JSBRDJBE.js";

// src/content/index.ts
import { remark } from "remark";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import { toJsxRuntime } from "hast-util-to-jsx-runtime";
import * as JsxRuntime from "react/jsx-runtime";
function rehypeReact(options = {}) {
  this.compiler = (tree, file) => {
    return toJsxRuntime(tree, {
      development: false,
      filePath: file.path,
      ...JsxRuntime,
      ...options
    });
  };
}
async function Markdown({
  children: content,
  remarkPlugins = [],
  rehypePlugins = [],
  ...options
}) {
  const processor = remark().use(remarkGfm).use(remarkPlugins).use(remarkRehype).use(rehypePlugins).use(rehypeReact, options);
  return (await processor.process(content)).result;
}
export {
  Markdown
};
