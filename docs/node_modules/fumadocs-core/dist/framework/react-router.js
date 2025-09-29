import {
  FrameworkProvider
} from "../chunk-BBP7MIO4.js";
import "../chunk-JSBRDJBE.js";

// src/framework/react-router.tsx
import { useMemo } from "react";
import {
  Link,
  useLocation,
  useNavigate,
  useParams,
  useRevalidator
} from "react-router";
import { jsx } from "react/jsx-runtime";
var framework = {
  usePathname() {
    return useLocation().pathname;
  },
  useParams() {
    return useParams();
  },
  useRouter() {
    const navigate = useNavigate();
    const revalidator = useRevalidator();
    return useMemo(
      () => ({
        push(url) {
          navigate(url);
        },
        refresh() {
          void revalidator.revalidate();
        }
      }),
      [navigate, revalidator]
    );
  },
  Link({ href, prefetch, ...props }) {
    return /* @__PURE__ */ jsx(Link, { to: href, prefetch: prefetch ? "intent" : "none", ...props, children: props.children });
  }
};
function ReactRouterProvider({ children }) {
  return /* @__PURE__ */ jsx(FrameworkProvider, { ...framework, children });
}
export {
  ReactRouterProvider
};
