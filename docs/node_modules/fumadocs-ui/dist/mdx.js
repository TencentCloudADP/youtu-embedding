import { jsx as _jsx } from "react/jsx-runtime";
import Link from 'fumadocs-core/link';
import { Image as FrameworkImage } from 'fumadocs-core/framework';
import { Card, Cards } from './components/card.js';
import { Callout } from './components/callout.js';
import { Heading } from './components/heading.js';
import { cn } from './utils/cn.js';
import { CodeBlock, CodeBlockTab, CodeBlockTabs, CodeBlockTabsList, CodeBlockTabsTrigger, Pre, } from './components/codeblock.js';
function Image(props) {
    return (_jsx(FrameworkImage, { sizes: "(max-width: 768px) 100vw, (max-width: 1200px) 70vw, 900px", ...props, src: props.src, className: cn('rounded-lg', props.className) }));
}
function Table(props) {
    return (_jsx("div", { className: "relative overflow-auto prose-no-margin my-6", children: _jsx("table", { ...props }) }));
}
const defaultMdxComponents = {
    CodeBlockTab,
    CodeBlockTabs,
    CodeBlockTabsList,
    CodeBlockTabsTrigger,
    pre: (props) => (_jsx(CodeBlock, { ...props, children: _jsx(Pre, { children: props.children }) })),
    Card,
    Cards,
    a: Link,
    img: Image,
    h1: (props) => (_jsx(Heading, { as: "h1", ...props })),
    h2: (props) => (_jsx(Heading, { as: "h2", ...props })),
    h3: (props) => (_jsx(Heading, { as: "h3", ...props })),
    h4: (props) => (_jsx(Heading, { as: "h4", ...props })),
    h5: (props) => (_jsx(Heading, { as: "h5", ...props })),
    h6: (props) => (_jsx(Heading, { as: "h6", ...props })),
    table: Table,
    Callout,
};
export const createRelativeLink = () => {
    throw new Error('`createRelativeLink` is only supported in Node.js environment');
};
export { defaultMdxComponents as default };
