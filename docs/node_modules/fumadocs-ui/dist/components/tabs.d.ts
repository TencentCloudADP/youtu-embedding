import * as React from 'react';
import { type ComponentProps, type ReactNode } from 'react';
import * as Unstyled from './tabs.unstyled.js';
export interface TabsProps extends Omit<ComponentProps<typeof Unstyled.Tabs>, 'value' | 'onValueChange'> {
    /**
     * Use simple mode instead of advanced usage as documented in https://radix-ui.com/primitives/docs/components/tabs.
     */
    items?: string[];
    /**
     * Shortcut for `defaultValue` when `items` is provided.
     *
     * @defaultValue 0
     */
    defaultIndex?: number;
    /**
     * Additional label in tabs list when `items` is provided.
     */
    label?: ReactNode;
}
export declare const TabsList: React.ForwardRefExoticComponent<Omit<import("@radix-ui/react-tabs").TabsListProps & React.RefAttributes<HTMLDivElement>, "ref"> & React.RefAttributes<HTMLDivElement>>;
export declare const TabsTrigger: React.ForwardRefExoticComponent<Omit<import("@radix-ui/react-tabs").TabsTriggerProps & React.RefAttributes<HTMLButtonElement>, "ref"> & React.RefAttributes<HTMLButtonElement>>;
export declare function Tabs({ ref, className, items, label, defaultIndex, defaultValue, ...props }: TabsProps): import("react/jsx-runtime").JSX.Element;
export interface TabProps extends Omit<ComponentProps<typeof Unstyled.TabsContent>, 'value'> {
    /**
     * Value of tab, detect from index if unspecified.
     */
    value?: string;
}
export declare function Tab({ value, ...props }: TabProps): import("react/jsx-runtime").JSX.Element;
export declare function TabsContent({ value, className, ...props }: ComponentProps<typeof Unstyled.TabsContent>): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=tabs.d.ts.map