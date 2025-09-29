// src/runtime/vite/browser.ts
import { createElement, lazy } from "react";

// src/runtime/vite/base.ts
function fromConfigBase() {
  function normalize(entries, base) {
    const out = {};
    for (const k in entries) {
      const mappedK = k.startsWith("./") ? k.slice(2) : k;
      if (base) Object.assign(entries[k], { base });
      out[mappedK] = entries[k];
    }
    return out;
  }
  return {
    doc(_, base, glob) {
      return normalize(glob, base);
    },
    meta(_, base, glob) {
      return normalize(glob, base);
    },
    docLazy(_, base, head, body) {
      return {
        base,
        head: normalize(head),
        body: normalize(body)
      };
    }
  };
}

// src/runtime/vite/browser.ts
var loaderStore = /* @__PURE__ */ new Map();
function createClientLoader(files, options) {
  const { id = "", component } = options;
  let renderer;
  const store = loaderStore.get(id) ?? {
    preloaded: /* @__PURE__ */ new Map()
  };
  loaderStore.set(id, store);
  function getRenderer() {
    if (renderer) return renderer;
    renderer = {};
    for (const k in files) {
      const OnDemand = lazy(async () => {
        const loaded = await files[k]();
        return { default: (props) => component(loaded, props) };
      });
      renderer[k] = (props) => {
        const cached = store.preloaded.get(k);
        if (!cached) return createElement(OnDemand, props);
        return component(cached, props);
      };
    }
    return renderer;
  }
  return {
    async preload(path) {
      const loaded = await files[path]();
      store.preloaded.set(path, loaded);
      return loaded;
    },
    getRenderer,
    getComponent(path) {
      return getRenderer()[path];
    }
  };
}
function toClientRenderer(files, component) {
  return createClientLoader(files, { component }).getRenderer();
}

export {
  fromConfigBase,
  createClientLoader,
  toClientRenderer
};
