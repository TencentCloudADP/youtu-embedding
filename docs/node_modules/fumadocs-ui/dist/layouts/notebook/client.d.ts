import { type ComponentProps } from 'react';
import type { Option } from '../../components/layout/root-toggle.js';
export declare function Navbar({ mode, ...props }: ComponentProps<'header'> & {
    mode: 'top' | 'auto';
}): import("react/jsx-runtime").JSX.Element;
export declare function LayoutBody(props: ComponentProps<'main'>): import("react/jsx-runtime").JSX.Element;
export declare function NavbarSidebarTrigger({ className, ...props }: ComponentProps<'button'>): import("react/jsx-runtime").JSX.Element;
export declare function LayoutTabs({ options, ...props }: ComponentProps<'div'> & {
    options: Option[];
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=client.d.ts.map