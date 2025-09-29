"use client";
import "./chunk-JSBRDJBE.js";

// src/hide-if-empty.tsx
import {
  createContext,
  useContext,
  useEffect,
  useId,
  useMemo,
  useState
} from "react";
import { Fragment, jsx, jsxs } from "react/jsx-runtime";
var Context = createContext({
  nonce: void 0
});
function HideIfEmptyProvider({
  nonce,
  children
}) {
  return /* @__PURE__ */ jsx(Context.Provider, { value: useMemo(() => ({ nonce }), [nonce]), children });
}
function getElement(id) {
  return document.querySelector(`[data-fd-if-empty="${id}"]`);
}
function isEmpty(node) {
  for (let i = 0; i < node.childNodes.length; i++) {
    const child = node.childNodes.item(i);
    if (child.nodeType === Node.TEXT_NODE || child.nodeType === Node.ELEMENT_NODE && window.getComputedStyle(child).display !== "none") {
      return false;
    }
  }
  return true;
}
function HideIfEmpty({
  as: Comp,
  ...props
}) {
  const id = useId();
  const { nonce } = useContext(Context);
  const [empty, setEmpty] = useState(() => {
    const element = typeof window !== "undefined" ? getElement(id) : null;
    if (element) return isEmpty(element);
  });
  useEffect(() => {
    const handleResize = () => {
      const element = getElement(id);
      if (element) setEmpty(isEmpty(element));
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [id]);
  const init = (id2) => {
    const element = getElement(id2);
    if (element) element.hidden = isEmpty(element);
    const script = document.currentScript;
    if (script) script.parentNode?.removeChild(script);
  };
  return /* @__PURE__ */ jsxs(Fragment, { children: [
    /* @__PURE__ */ jsx(
      Comp,
      {
        ...props,
        "data-fd-if-empty": id,
        hidden: empty ?? false
      }
    ),
    empty === void 0 && /* @__PURE__ */ jsx(
      "script",
      {
        nonce,
        dangerouslySetInnerHTML: {
          __html: `{${getElement};${isEmpty};(${init})("${id}")}`
        }
      }
    )
  ] });
}
export {
  HideIfEmpty,
  HideIfEmptyProvider
};
