"use client";
import "../chunk-JSBRDJBE.js";

// src/utils/use-effect-event.ts
import { useCallback, useRef } from "react";
function useEffectEvent(callback) {
  const ref = useRef(callback);
  ref.current = callback;
  return useCallback(((...params) => ref.current(...params)), []);
}
export {
  useEffectEvent
};
