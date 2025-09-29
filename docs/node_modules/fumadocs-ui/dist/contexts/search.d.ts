import { type ComponentType, type ReactNode } from 'react';
interface HotKey {
    display: ReactNode;
    /**
     * Key code or a function determining whether the key is pressed.
     */
    key: string | ((e: KeyboardEvent) => boolean);
}
export interface SharedProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}
export type SearchLink = [name: string, href: string];
export interface TagItem {
    name: string;
    value: string;
}
export interface SearchProviderProps {
    /**
     * Preload search dialog before opening it
     *
     * @defaultValue `true`
     */
    preload?: boolean;
    /**
     * Custom links to be displayed if search is empty
     */
    links?: SearchLink[];
    /**
     * Hotkeys for triggering search dialog
     *
     * @defaultValue Meta/Ctrl + K
     */
    hotKey?: HotKey[];
    /**
     * Replace default search dialog, allowing you to use other solutions such as Algolia Search
     *
     * It receives the `open` and `onOpenChange` prop, can be lazy loaded with `next/dynamic`
     */
    SearchDialog: ComponentType<SharedProps>;
    /**
     * Additional props to the dialog
     */
    options?: Partial<SharedProps & Record<string, unknown>>;
    children?: ReactNode;
}
interface SearchContextType {
    enabled: boolean;
    hotKey: HotKey[];
    setOpenSearch: (value: boolean) => void;
}
export declare function useSearchContext(): SearchContextType;
export declare function SearchProvider({ SearchDialog, children, preload, options, hotKey, links, }: SearchProviderProps): import("react/jsx-runtime").JSX.Element;
/**
 * Show children only when search is enabled via React Context
 */
export declare function SearchOnly({ children }: {
    children: ReactNode;
}): ReactNode;
export {};
//# sourceMappingURL=search.d.ts.map