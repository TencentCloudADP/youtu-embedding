"use client";
import {
  useOnChange
} from "./chunk-EMWGTXSW.js";
import "./chunk-JSBRDJBE.js";

// src/toc.tsx
import { createContext, forwardRef, useContext, useMemo, useRef } from "react";
import scrollIntoView from "scroll-into-view-if-needed";

// src/utils/merge-refs.ts
function mergeRefs(...refs) {
  return (value) => {
    refs.forEach((ref) => {
      if (typeof ref === "function") {
        ref(value);
      } else if (ref !== null) {
        ref.current = value;
      }
    });
  };
}

// src/utils/use-anchor-observer.ts
import { useEffect, useState } from "react";
function useAnchorObserver(watch, single) {
  const [activeAnchor, setActiveAnchor] = useState([]);
  useEffect(() => {
    let visible = [];
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && !visible.includes(entry.target.id)) {
            visible = [...visible, entry.target.id];
          } else if (!entry.isIntersecting && visible.includes(entry.target.id)) {
            visible = visible.filter((v) => v !== entry.target.id);
          }
        }
        if (visible.length > 0) setActiveAnchor(visible);
      },
      {
        rootMargin: single ? "-80px 0% -70% 0%" : `-20px 0% -40% 0%`,
        threshold: 1
      }
    );
    function onScroll() {
      const element = document.scrollingElement;
      if (!element) return;
      const top = element.scrollTop;
      if (top <= 0 && single) setActiveAnchor(watch.slice(0, 1));
      else if (top + element.clientHeight >= element.scrollHeight - 6) {
        setActiveAnchor((active) => {
          return active.length > 0 && !single ? watch.slice(watch.indexOf(active[0])) : watch.slice(-1);
        });
      }
    }
    for (const heading of watch) {
      const element = document.getElementById(heading);
      if (element) observer.observe(element);
    }
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      observer.disconnect();
    };
  }, [single, watch]);
  return single ? activeAnchor.slice(0, 1) : activeAnchor;
}

// src/toc.tsx
import { jsx } from "react/jsx-runtime";
var ActiveAnchorContext = createContext([]);
var ScrollContext = createContext({
  current: null
});
function useActiveAnchor() {
  return useContext(ActiveAnchorContext).at(-1);
}
function useActiveAnchors() {
  return useContext(ActiveAnchorContext);
}
function ScrollProvider({
  containerRef,
  children
}) {
  return /* @__PURE__ */ jsx(ScrollContext.Provider, { value: containerRef, children });
}
function AnchorProvider({
  toc,
  single = true,
  children
}) {
  const headings = useMemo(() => {
    return toc.map((item) => item.url.split("#")[1]);
  }, [toc]);
  return /* @__PURE__ */ jsx(ActiveAnchorContext.Provider, { value: useAnchorObserver(headings, single), children });
}
var TOCItem = forwardRef(
  ({ onActiveChange, ...props }, ref) => {
    const containerRef = useContext(ScrollContext);
    const anchors = useActiveAnchors();
    const anchorRef = useRef(null);
    const mergedRef = mergeRefs(anchorRef, ref);
    const isActive = anchors.includes(props.href.slice(1));
    useOnChange(isActive, (v) => {
      const element = anchorRef.current;
      if (!element) return;
      if (v && containerRef.current) {
        scrollIntoView(element, {
          behavior: "smooth",
          block: "center",
          inline: "center",
          scrollMode: "always",
          boundary: containerRef.current
        });
      }
      onActiveChange?.(v);
    });
    return /* @__PURE__ */ jsx("a", { ref: mergedRef, "data-active": isActive, ...props, children: props.children });
  }
);
TOCItem.displayName = "TOCItem";
export {
  AnchorProvider,
  ScrollProvider,
  TOCItem,
  useActiveAnchor,
  useActiveAnchors
};
