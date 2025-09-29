'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { cva } from 'class-variance-authority';
import { File as FileIcon, Folder as FolderIcon, FolderOpen, } from '../icons.js';
import { useState } from 'react';
import { cn } from '../utils/cn.js';
import { Collapsible, CollapsibleContent, CollapsibleTrigger, } from './ui/collapsible.js';
const itemVariants = cva('flex flex-row items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-fd-accent hover:text-fd-accent-foreground [&_svg]:size-4');
export function Files({ className, ...props }) {
    return (_jsx("div", { className: cn('not-prose rounded-md border bg-fd-card p-2', className), ...props, children: props.children }));
}
export function File({ name, icon = _jsx(FileIcon, {}), className, ...rest }) {
    return (_jsxs("div", { className: cn(itemVariants({ className })), ...rest, children: [icon, name] }));
}
export function Folder({ name, defaultOpen = false, ...props }) {
    const [open, setOpen] = useState(defaultOpen);
    return (_jsxs(Collapsible, { open: open, onOpenChange: setOpen, ...props, children: [_jsxs(CollapsibleTrigger, { className: cn(itemVariants({ className: 'w-full' })), children: [open ? _jsx(FolderOpen, {}) : _jsx(FolderIcon, {}), name] }), _jsx(CollapsibleContent, { children: _jsx("div", { className: "ms-2 flex flex-col border-l ps-2", children: props.children }) })] }));
}
