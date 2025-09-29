import type { AccordionMultipleProps, AccordionSingleProps } from '@radix-ui/react-accordion';
import * as AccordionPrimitive from '@radix-ui/react-accordion';
import { type ReactNode } from 'react';
export declare const Accordions: import("react").ForwardRefExoticComponent<(Omit<AccordionSingleProps, "value" | "onValueChange"> | Omit<AccordionMultipleProps, "value" | "onValueChange">) & import("react").RefAttributes<HTMLDivElement>>;
export declare const Accordion: import("react").ForwardRefExoticComponent<Omit<Omit<AccordionPrimitive.AccordionItemProps & import("react").RefAttributes<HTMLDivElement>, "ref">, "title" | "value"> & {
    title: string | ReactNode;
    value?: string;
} & import("react").RefAttributes<HTMLDivElement>>;
//# sourceMappingURL=accordion.d.ts.map