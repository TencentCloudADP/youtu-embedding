"use client";
import {
  Link
} from "./chunk-H2GMUTQG.js";
import {
  useParams
} from "./chunk-BBP7MIO4.js";
import "./chunk-JSBRDJBE.js";

// src/dynamic-link.tsx
import { forwardRef, useMemo } from "react";
import { jsx } from "react/jsx-runtime";
var DynamicLink = forwardRef(
  ({ href, ...props }, ref) => {
    const params = useParams();
    const url = useMemo(() => {
      return href?.replace(/\[.*]/, (key) => {
        const mappedKey = key.slice(1, -1);
        const value = mappedKey in params ? params[mappedKey] : void 0;
        if (!value) return "";
        return typeof value === "string" ? value : value.join("/");
      });
    }, [params, href]);
    return /* @__PURE__ */ jsx(Link, { ref, href: url, ...props });
  }
);
DynamicLink.displayName = "DynamicLink";
var dynamic_link_default = DynamicLink;
export {
  DynamicLink,
  dynamic_link_default as default
};
