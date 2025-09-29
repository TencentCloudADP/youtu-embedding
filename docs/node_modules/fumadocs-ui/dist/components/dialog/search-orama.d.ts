import { type OramaCloudOptions } from 'fumadocs-core/search/client';
import { type ReactNode } from 'react';
import { type SharedProps } from './search.js';
import type { SearchLink, TagItem } from '../../contexts/search.js';
export interface OramaSearchDialogProps extends SharedProps {
    links?: SearchLink[];
    client: OramaCloudOptions['client'];
    searchOptions?: OramaCloudOptions['params'];
    index?: OramaCloudOptions['index'];
    footer?: ReactNode;
    defaultTag?: string;
    tags?: TagItem[];
    /**
     * Add the "Powered by Orama" label
     *
     * @defaultValue false
     */
    showOrama?: boolean;
    /**
     * Allow to clear tag filters
     *
     * @defaultValue false
     */
    allowClear?: boolean;
}
/**
 * Orama Cloud integration
 */
export default function OramaSearchDialog({ client, searchOptions, tags, defaultTag, showOrama, allowClear, index, footer, links, ...props }: OramaSearchDialogProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=search-orama.d.ts.map