// src/framework/index.tsx
import React from "react";
import { jsx } from "react/jsx-runtime";
var notImplemented = () => {
  throw new Error(
    "You need to wrap your application inside `FrameworkProvider`."
  );
};
var FrameworkContext = createContext("FrameworkContext", {
  useParams: notImplemented,
  useRouter: notImplemented,
  usePathname: notImplemented
});
function FrameworkProvider({
  Link: Link2,
  useRouter: useRouter2,
  useParams: useParams2,
  usePathname: usePathname2,
  Image: Image2,
  children
}) {
  const framework = React.useMemo(
    () => ({
      usePathname: usePathname2,
      useRouter: useRouter2,
      Link: Link2,
      Image: Image2,
      useParams: useParams2
    }),
    [Link2, usePathname2, useRouter2, useParams2, Image2]
  );
  return /* @__PURE__ */ jsx(FrameworkContext.Provider, { value: framework, children });
}
function usePathname() {
  return FrameworkContext.use().usePathname();
}
function useRouter() {
  return FrameworkContext.use().useRouter();
}
function useParams() {
  return FrameworkContext.use().useParams();
}
function Image(props) {
  const { Image: Image2 } = FrameworkContext.use();
  if (!Image2) {
    const { src, alt, priority, ...rest } = props;
    return /* @__PURE__ */ jsx(
      "img",
      {
        alt,
        src,
        fetchPriority: priority ? "high" : "auto",
        ...rest
      }
    );
  }
  return /* @__PURE__ */ jsx(Image2, { ...props });
}
function Link(props) {
  const { Link: Link2 } = FrameworkContext.use();
  if (!Link2) {
    const { href, prefetch: _, ...rest } = props;
    return /* @__PURE__ */ jsx("a", { href, ...rest });
  }
  return /* @__PURE__ */ jsx(Link2, { ...props });
}
function createContext(name, v) {
  const Context = React.createContext(v);
  return {
    Provider: (props) => {
      return /* @__PURE__ */ jsx(Context.Provider, { value: props.value, children: props.children });
    },
    use: (errorMessage) => {
      const value = React.useContext(Context);
      if (!value)
        throw new Error(
          errorMessage ?? `Provider of ${name} is required but missing.`
        );
      return value;
    }
  };
}

export {
  FrameworkProvider,
  usePathname,
  useRouter,
  useParams,
  Image,
  Link,
  createContext
};
