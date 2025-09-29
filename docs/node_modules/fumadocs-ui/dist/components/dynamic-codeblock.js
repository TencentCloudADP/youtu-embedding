'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { CodeBlock, Pre } from '../components/codeblock.js';
import { useShiki } from 'fumadocs-core/highlight/client';
import { cn } from '../utils/cn.js';
import { createContext, Suspense, use, } from 'react';
const PropsContext = createContext(undefined);
function DefaultPre(props) {
    const extraProps = use(PropsContext);
    return (_jsx(CodeBlock, { ...props, ...extraProps, className: cn('my-0', props.className, extraProps?.className), children: _jsx(Pre, { children: props.children }) }));
}
export function DynamicCodeBlock({ lang, code, codeblock, options, wrapInSuspense = true, }) {
    const shikiOptions = {
        lang,
        ...options,
        components: {
            pre: DefaultPre,
            ...options?.components,
        },
    };
    let children = _jsx(Internal, { code: code, options: shikiOptions });
    if (wrapInSuspense)
        children = (_jsx(Suspense, { fallback: _jsx(Placeholder, { code: code, components: shikiOptions.components }), children: children }));
    return _jsx(PropsContext, { value: codeblock, children: children });
}
function Placeholder({ code, components = {}, }) {
    const { pre: Pre = 'pre', code: Code = 'code' } = components;
    return (_jsx(Pre, { children: _jsx(Code, { children: code.split('\n').map((line, i) => (_jsx("span", { className: "line", children: line }, i))) }) }));
}
function Internal({ code, options, }) {
    return useShiki(code, options);
}
