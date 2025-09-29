import { type AlgoliaOptions } from 'fumadocs-core/search/client';
import { type ReactNode } from 'react';
import { type SharedProps } from './search.js';
import type { SearchLink, TagItem } from '../../contexts/search.js';
export interface AlgoliaSearchDialogProps extends SharedProps {
    searchOptions: AlgoliaOptions;
    links?: SearchLink[];
    footer?: ReactNode;
    defaultTag?: string;
    tags?: TagItem[];
    /**
     * Add the "Powered by Algolia" label, this is useful for free tier users
     *
     * @defaultValue false
     */
    showAlgolia?: boolean;
    /**
     * Allow to clear tag filters
     *
     * @defaultValue false
     */
    allowClear?: boolean;
}
export default function AlgoliaSearchDialog({ searchOptions, tags, defaultTag, showAlgolia, allowClear, links, footer, ...props }: AlgoliaSearchDialogProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=search-algolia.d.ts.map