'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useDocsSearch, } from 'fumadocs-core/search/client';
import { useMemo, useState } from 'react';
import { useOnChange } from 'fumadocs-core/utils/use-on-change';
import { SearchDialog, SearchDialogClose, SearchDialogContent, SearchDialogFooter, SearchDialogHeader, SearchDialogIcon, SearchDialogInput, SearchDialogList, SearchDialogOverlay, TagsList, TagsListItem, } from './search.js';
import { useI18n } from '../../contexts/i18n.js';
/**
 * Orama Cloud integration
 */
export default function OramaSearchDialog({ client, searchOptions, tags = [], defaultTag, showOrama = false, allowClear = false, index, footer, links = [], ...props }) {
    const { locale } = useI18n();
    const [tag, setTag] = useState(defaultTag);
    const { search, setSearch, query } = useDocsSearch({
        type: 'orama-cloud',
        client,
        index,
        params: searchOptions,
        locale,
        tag,
    });
    const defaultItems = useMemo(() => {
        if (links.length === 0)
            return null;
        return links.map(([name, link]) => ({
            type: 'page',
            id: name,
            content: name,
            url: link,
        }));
    }, [links]);
    useOnChange(defaultTag, (v) => {
        setTag(v);
    });
    const label = showOrama && _jsx(Label, {});
    return (_jsxs(SearchDialog, { search: search, onSearchChange: setSearch, isLoading: query.isLoading, ...props, children: [_jsx(SearchDialogOverlay, {}), _jsxs(SearchDialogContent, { children: [_jsxs(SearchDialogHeader, { children: [_jsx(SearchDialogIcon, {}), _jsx(SearchDialogInput, {}), _jsx(SearchDialogClose, {})] }), _jsx(SearchDialogList, { items: query.data !== 'empty' ? query.data : defaultItems }), _jsxs(SearchDialogFooter, { children: [tags.length > 0 ? (_jsxs(TagsList, { tag: tag, onTagChange: setTag, allowClear: allowClear, children: [tags.map((tag) => (_jsx(TagsListItem, { value: tag.value, children: tag.name }, tag.value))), label] })) : (label), footer] })] })] }));
}
function Label() {
    return (_jsx("a", { href: "https://orama.com", rel: "noreferrer noopener", className: "ms-auto text-xs text-fd-muted-foreground", children: "Search powered by Orama" }));
}
