import { type ReactNode } from 'react';
export interface PageStyles {
    tocNav?: string;
    toc?: string;
    page?: string;
    article?: string;
}
export declare function usePageStyles(): PageStyles;
export declare function StylesProvider({ children, ...value }: PageStyles & {
    children: ReactNode;
}): import("react/jsx-runtime").JSX.Element;
export interface NavProviderProps {
    /**
     * Use transparent background
     *
     * @defaultValue none
     */
    transparentMode?: 'always' | 'top' | 'none';
}
interface NavContextType {
    isTransparent: boolean;
}
export declare function NavProvider({ transparentMode, children, }: NavProviderProps & {
    children: ReactNode;
}): import("react/jsx-runtime").JSX.Element;
export declare function useNav(): NavContextType;
export {};
//# sourceMappingURL=layout.d.ts.map