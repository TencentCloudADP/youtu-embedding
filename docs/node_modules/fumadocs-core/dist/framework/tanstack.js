import {
  FrameworkProvider
} from "../chunk-BBP7MIO4.js";
import "../chunk-JSBRDJBE.js";

// src/framework/tanstack.tsx
import { useMemo } from "react";
import {
  useParams,
  Link,
  useRouter,
  useLocation
} from "@tanstack/react-router";
import { jsx } from "react/jsx-runtime";
var framework = {
  Link({ href, prefetch, ...props }) {
    return /* @__PURE__ */ jsx(Link, { to: href, preload: prefetch ? "intent" : false, ...props, children: props.children });
  },
  usePathname() {
    return useLocation().pathname;
  },
  useRouter() {
    const router = useRouter();
    return useMemo(
      () => ({
        push(url) {
          void router.navigate({
            href: url
          });
        },
        refresh() {
          void router.invalidate();
        }
      }),
      [router]
    );
  },
  useParams() {
    return useParams({ strict: false });
  }
};
function TanstackProvider({ children }) {
  return /* @__PURE__ */ jsx(FrameworkProvider, { ...framework, children });
}
export {
  TanstackProvider
};
