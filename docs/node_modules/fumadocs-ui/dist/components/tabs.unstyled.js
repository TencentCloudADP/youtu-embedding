'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useLayoutEffect, useMemo, useRef, useState, } from 'react';
import * as Primitive from '@radix-ui/react-tabs';
import { mergeRefs } from '../utils/merge-refs.js';
import { useEffectEvent } from 'fumadocs-core/utils/use-effect-event';
const listeners = new Map();
function addChangeListener(id, listener) {
    const list = listeners.get(id) ?? [];
    list.push(listener);
    listeners.set(id, list);
}
function removeChangeListener(id, listener) {
    const list = listeners.get(id) ?? [];
    listeners.set(id, list.filter((item) => item !== listener));
}
const TabsContext = createContext(null);
function useTabContext() {
    const ctx = useContext(TabsContext);
    if (!ctx)
        throw new Error('You must wrap your component in <Tabs>');
    return ctx;
}
export const TabsList = Primitive.TabsList;
export const TabsTrigger = Primitive.TabsTrigger;
/**
 * @internal You better not use it
 */
export function Tabs({ ref, groupId, persist = false, updateAnchor = false, defaultValue, value: _value, onValueChange: _onValueChange, ...props }) {
    const tabsRef = useRef(null);
    const [value, setValue] = _value === undefined
        ? // eslint-disable-next-line react-hooks/rules-of-hooks -- not supposed to change controlled/uncontrolled
            useState(defaultValue)
        : [_value, _onValueChange ?? (() => undefined)];
    const onChange = useEffectEvent((v) => setValue(v));
    const valueToIdMap = useMemo(() => new Map(), []);
    useLayoutEffect(() => {
        if (!groupId)
            return;
        const previous = persist
            ? localStorage.getItem(groupId)
            : sessionStorage.getItem(groupId);
        if (previous)
            onChange(previous);
        addChangeListener(groupId, onChange);
        return () => {
            removeChangeListener(groupId, onChange);
        };
    }, [groupId, onChange, persist]);
    useLayoutEffect(() => {
        const hash = window.location.hash.slice(1);
        if (!hash)
            return;
        for (const [value, id] of valueToIdMap.entries()) {
            if (id === hash) {
                onChange(value);
                tabsRef.current?.scrollIntoView();
                break;
            }
        }
    }, [onChange, valueToIdMap]);
    return (_jsx(Primitive.Tabs, { ref: mergeRefs(ref, tabsRef), value: value, onValueChange: (v) => {
            if (updateAnchor) {
                const id = valueToIdMap.get(v);
                if (id) {
                    window.history.replaceState(null, '', `#${id}`);
                }
            }
            if (groupId) {
                listeners.get(groupId)?.forEach((item) => {
                    item(v);
                });
                if (persist)
                    localStorage.setItem(groupId, v);
                else
                    sessionStorage.setItem(groupId, v);
            }
            else {
                setValue(v);
            }
        }, ...props, children: _jsx(TabsContext.Provider, { value: useMemo(() => ({ valueToIdMap }), [valueToIdMap]), children: props.children }) }));
}
export function TabsContent({ value, ...props }) {
    const { valueToIdMap } = useTabContext();
    if (props.id) {
        valueToIdMap.set(value, props.id);
    }
    return (_jsx(Primitive.TabsContent, { value: value, ...props, children: props.children }));
}
