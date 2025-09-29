"use client";
import {
  highlight
} from "../chunk-HN2NUUD2.js";
import "../chunk-JSBRDJBE.js";

// src/highlight/client.tsx
import {
  use,
  useId,
  useMemo
} from "react";
var promises = {};
function useShiki(code, options, deps) {
  const id = useId();
  const key = useMemo(() => {
    const state = deps ? JSON.stringify(deps) : `${options.lang}:${code}`;
    return `${id}:${state}`;
  }, [code, deps, id, options.lang]);
  return use(
    promises[key] ??= highlight(code, {
      ...options,
      engine: options.engine ?? "js"
    })
  );
}
export {
  useShiki
};
