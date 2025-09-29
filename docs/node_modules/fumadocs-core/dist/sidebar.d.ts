import * as react_jsx_runtime from 'react/jsx-runtime';
import { ReactNode, ElementType, ComponentPropsWithoutRef, ReactElement } from 'react';

interface SidebarProviderProps {
    open?: boolean;
    onOpenChange?: (v: boolean) => void;
    children: ReactNode;
}
declare function SidebarProvider(props: SidebarProviderProps): react_jsx_runtime.JSX.Element;
type AsProps<T extends ElementType> = Omit<ComponentPropsWithoutRef<T>, 'as'> & {
    as?: T;
};
type SidebarTriggerProps<T extends ElementType> = AsProps<T>;
declare function SidebarTrigger<T extends ElementType = 'button'>({ as, ...props }: SidebarTriggerProps<T>): ReactElement;
type SidebarContentProps<T extends ElementType> = AsProps<T> & {
    /**
     * Disable scroll blocking when the viewport width is larger than a certain number (in pixels)
     *
     * @deprecated use `removeScrollOn`
     */
    blockScrollingWidth?: number;
    /**
     * A media query.
     *
     * When the sidebar is opening and media query is matched, scrolling outside the sidebar will be blocked.
     *
     * @example (max-width: 1000px)
     */
    removeScrollOn?: string;
};
declare function SidebarList<T extends ElementType = 'aside'>({ as, blockScrollingWidth, removeScrollOn, ...props }: SidebarContentProps<T>): ReactElement;

export { type SidebarContentProps, SidebarList, SidebarProvider, type SidebarProviderProps, SidebarTrigger, type SidebarTriggerProps };
