import { type ReactNode } from 'react';
import { type SharedProps } from './search.js';
import type { SearchLink, TagItem } from '../../contexts/search.js';
export interface DefaultSearchDialogProps extends SharedProps {
    links?: SearchLink[];
    /**
     * @defaultValue 'fetch'
     */
    type?: 'fetch' | 'static';
    defaultTag?: string;
    tags?: TagItem[];
    /**
     * Search API URL
     */
    api?: string;
    /**
     * The debounced delay for performing a search.
     */
    delayMs?: number;
    footer?: ReactNode;
    /**
     * Allow to clear tag filters
     *
     * @defaultValue false
     */
    allowClear?: boolean;
}
export default function DefaultSearchDialog({ defaultTag, tags, api, delayMs, type, allowClear, links, footer, ...props }: DefaultSearchDialogProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=search-default.d.ts.map