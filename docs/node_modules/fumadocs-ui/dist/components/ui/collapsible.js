'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import * as CollapsiblePrimitive from '@radix-ui/react-collapsible';
import { forwardRef, useEffect, useState } from 'react';
import { cn } from '../../utils/cn.js';
const Collapsible = CollapsiblePrimitive.Root;
const CollapsibleTrigger = CollapsiblePrimitive.CollapsibleTrigger;
const CollapsibleContent = forwardRef(({ children, ...props }, ref) => {
    const [mounted, setMounted] = useState(false);
    useEffect(() => {
        setMounted(true);
    }, []);
    return (_jsx(CollapsiblePrimitive.CollapsibleContent, { ref: ref, ...props, className: cn('overflow-hidden', mounted &&
            'data-[state=closed]:animate-fd-collapsible-up data-[state=open]:animate-fd-collapsible-down', props.className), children: children }));
});
CollapsibleContent.displayName =
    CollapsiblePrimitive.CollapsibleContent.displayName;
export { Collapsible, CollapsibleTrigger, CollapsibleContent };
