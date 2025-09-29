"use client";
import {
  useMediaQuery
} from "./chunk-EP5LHGDZ.js";
import "./chunk-JSBRDJBE.js";

// src/sidebar.tsx
import {
  createContext,
  useContext,
  useMemo,
  useState
} from "react";
import { RemoveScroll } from "react-remove-scroll";
import { jsx } from "react/jsx-runtime";
var SidebarContext = createContext(null);
function useSidebarContext() {
  const ctx = useContext(SidebarContext);
  if (!ctx) throw new Error("Missing sidebar provider");
  return ctx;
}
function SidebarProvider(props) {
  const [open, setOpen] = props.open === void 0 ? (
    // eslint-disable-next-line react-hooks/rules-of-hooks
    useState(false)
  ) : [props.open, props.onOpenChange];
  return /* @__PURE__ */ jsx(
    SidebarContext.Provider,
    {
      value: useMemo(
        () => ({ open, setOpen: setOpen ?? (() => void 0) }),
        [open, setOpen]
      ),
      children: props.children
    }
  );
}
function SidebarTrigger({
  as,
  ...props
}) {
  const { open, setOpen } = useSidebarContext();
  const As = as ?? "button";
  return /* @__PURE__ */ jsx(
    As,
    {
      "aria-label": "Toggle Sidebar",
      "data-open": open,
      onClick: () => {
        setOpen(!open);
      },
      ...props
    }
  );
}
function SidebarList({
  as,
  blockScrollingWidth,
  removeScrollOn = blockScrollingWidth ? `(width < ${blockScrollingWidth}px)` : void 0,
  ...props
}) {
  const { open } = useSidebarContext();
  const isBlocking = useMediaQuery(removeScrollOn ?? "", !removeScrollOn) ?? false;
  return /* @__PURE__ */ jsx(
    RemoveScroll,
    {
      as: as ?? "aside",
      "data-open": open,
      enabled: isBlocking && open,
      ...props,
      children: props.children
    }
  );
}
export {
  SidebarList,
  SidebarProvider,
  SidebarTrigger
};
