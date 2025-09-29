"use client";
import {
  FrameworkProvider
} from "../chunk-BBP7MIO4.js";
import "../chunk-JSBRDJBE.js";

// src/framework/next.tsx
import { useParams, usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { jsx } from "react/jsx-runtime";
function NextProvider({ children }) {
  return /* @__PURE__ */ jsx(
    FrameworkProvider,
    {
      usePathname,
      useRouter,
      useParams,
      Link,
      Image,
      children
    }
  );
}
export {
  NextProvider
};
