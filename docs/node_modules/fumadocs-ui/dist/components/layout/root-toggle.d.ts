import { type ComponentProps, type ReactNode } from 'react';
import type { SidebarTab } from '../../utils/get-sidebar-tabs.js';
export interface Option extends SidebarTab {
    props?: ComponentProps<'a'>;
}
export declare function RootToggle({ options, placeholder, ...props }: {
    placeholder?: ReactNode;
    options: Option[];
} & ComponentProps<'button'>): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=root-toggle.d.ts.map