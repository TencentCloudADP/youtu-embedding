'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useDocsSearch, } from 'fumadocs-core/search/client';
import { useMemo, useState } from 'react';
import { useOnChange } from 'fumadocs-core/utils/use-on-change';
import { SearchDialog, SearchDialogClose, SearchDialogContent, SearchDialogFooter, SearchDialogHeader, SearchDialogIcon, SearchDialogInput, SearchDialogList, SearchDialogOverlay, TagsList, TagsListItem, } from './search.js';
import { useI18n } from '../../contexts/i18n.js';
export default function AlgoliaSearchDialog({ searchOptions, tags = [], defaultTag, showAlgolia = false, allowClear = false, links = [], footer, ...props }) {
    const [tag, setTag] = useState(defaultTag);
    const { locale } = useI18n();
    const { search, setSearch, query } = useDocsSearch({
        type: 'algolia',
        tag,
        locale,
        ...searchOptions,
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
    const label = showAlgolia && _jsx(AlgoliaTitle, {});
    return (_jsxs(SearchDialog, { search: search, onSearchChange: setSearch, isLoading: query.isLoading, ...props, children: [_jsx(SearchDialogOverlay, {}), _jsxs(SearchDialogContent, { children: [_jsxs(SearchDialogHeader, { children: [_jsx(SearchDialogIcon, {}), _jsx(SearchDialogInput, {}), _jsx(SearchDialogClose, {})] }), _jsx(SearchDialogList, { items: query.data !== 'empty' ? query.data : defaultItems })] }), _jsxs(SearchDialogFooter, { children: [tags.length > 0 ? (_jsxs(TagsList, { tag: tag, onTagChange: setTag, allowClear: allowClear, children: [tags.map((tag) => (_jsx(TagsListItem, { value: tag.value, children: tag.name }, tag.value))), label] })) : (label), footer] })] }));
}
function AlgoliaTitle() {
    return (_jsx("a", { href: "https://algolia.com", rel: "noreferrer noopener", className: "ms-auto text-xs text-fd-muted-foreground", children: "Search powered by Algolia" }));
}
