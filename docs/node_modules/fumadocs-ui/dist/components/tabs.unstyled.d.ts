import { type ComponentProps } from 'react';
import * as Primitive from '@radix-ui/react-tabs';
export interface TabsProps extends ComponentProps<typeof Primitive.Tabs> {
    /**
     * Identifier for Sharing value of tabs
     */
    groupId?: string;
    /**
     * Enable persistent
     */
    persist?: boolean;
    /**
     * If true, updates the URL hash based on the tab's id
     */
    updateAnchor?: boolean;
}
export declare const TabsList: import("react").ForwardRefExoticComponent<Primitive.TabsListProps & import("react").RefAttributes<HTMLDivElement>>;
export declare const TabsTrigger: import("react").ForwardRefExoticComponent<Primitive.TabsTriggerProps & import("react").RefAttributes<HTMLButtonElement>>;
/**
 * @internal You better not use it
 */
export declare function Tabs({ ref, groupId, persist, updateAnchor, defaultValue, value: _value, onValueChange: _onValueChange, ...props }: TabsProps): import("react/jsx-runtime").JSX.Element;
export declare function TabsContent({ value, ...props }: ComponentProps<typeof Primitive.TabsContent>): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=tabs.unstyled.d.ts.map