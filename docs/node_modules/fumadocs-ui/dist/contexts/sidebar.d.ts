import { type ReactNode, type RefObject } from 'react';
interface SidebarContext {
    open: boolean;
    setOpen: React.Dispatch<React.SetStateAction<boolean>>;
    collapsed: boolean;
    setCollapsed: React.Dispatch<React.SetStateAction<boolean>>;
    /**
     * When set to false, don't close the sidebar when navigate to another page
     */
    closeOnRedirect: RefObject<boolean>;
}
declare const SidebarContext: {
    Provider: (props: {
        value: SidebarContext;
        children: ReactNode;
    }) => import("react/jsx-runtime").JSX.Element;
    use: (errorMessage?: string) => SidebarContext;
};
export declare function useSidebar(): SidebarContext;
export declare function SidebarProvider({ children, }: {
    children: ReactNode;
}): ReactNode;
export {};
//# sourceMappingURL=sidebar.d.ts.map