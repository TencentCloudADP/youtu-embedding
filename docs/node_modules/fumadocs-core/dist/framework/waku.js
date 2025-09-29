"use client";
import {
  FrameworkProvider
} from "../chunk-BBP7MIO4.js";
import "../chunk-JSBRDJBE.js";

// src/framework/waku.tsx
import { useMemo } from "react";
import { Link as WakuLink, useRouter } from "waku";
import { jsx } from "react/jsx-runtime";
var framework = {
  usePathname() {
    const { path } = useRouter();
    return path;
  },
  useParams() {
    const { query } = useRouter();
    return useMemo(() => {
      const params = new URLSearchParams(query);
      return Object.fromEntries(
        Array.from(params.entries()).map(([key, value]) => [
          key,
          Array.isArray(value) ? value[0] : value
        ])
      );
    }, [query]);
  },
  useRouter() {
    const router = useRouter();
    return useMemo(
      () => ({
        push(url) {
          void router.push(url);
        },
        refresh() {
          void router.push(router.path);
        }
      }),
      [router]
    );
  },
  Link({ href, prefetch: _prefetch, ...props }) {
    return /* @__PURE__ */ jsx(WakuLink, { to: href, ...props, children: props.children });
  }
};
function WakuProvider({ children }) {
  return /* @__PURE__ */ jsx(FrameworkProvider, { ...framework, children });
}
export {
  WakuProvider
};
