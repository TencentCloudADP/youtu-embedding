'use client';
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useDocsSearch } from 'fumadocs-core/search/client';
import { useMemo, useState } from 'react';
import { useOnChange } from 'fumadocs-core/utils/use-on-change';
import { useI18n } from '../../contexts/i18n.js';
import { SearchDialog, SearchDialogClose, SearchDialogContent, SearchDialogFooter, SearchDialogHeader, SearchDialogIcon, SearchDialogInput, SearchDialogList, SearchDialogOverlay, TagsList, TagsListItem, } from './search.js';
export default function DefaultSearchDialog({ defaultTag, tags = [], api, delayMs, type = 'fetch', allowClear = false, links = [], footer, ...props }) {
    const { locale } = useI18n();
    const [tag, setTag] = useState(defaultTag);
    const { search, setSearch, query } = useDocsSearch(type === 'fetch'
        ? {
            type: 'fetch',
            api,
            locale,
            tag,
            delayMs,
        }
        : {
            type: 'static',
            from: api,
            locale,
            tag,
            delayMs,
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
    return (_jsxs(SearchDialog, { search: search, onSearchChange: setSearch, isLoading: query.isLoading, ...props, children: [_jsx(SearchDialogOverlay, {}), _jsxs(SearchDialogContent, { children: [_jsxs(SearchDialogHeader, { children: [_jsx(SearchDialogIcon, {}), _jsx(SearchDialogInput, {}), _jsx(SearchDialogClose, {})] }), _jsx(SearchDialogList, { items: query.data !== 'empty' ? query.data : defaultItems })] }), _jsxs(SearchDialogFooter, { children: [tags.length > 0 && (_jsx(TagsList, { tag: tag, onTagChange: setTag, allowClear: allowClear, children: tags.map((tag) => (_jsx(TagsListItem, { value: tag.value, children: tag.name }, tag.value))) })), footer] })] }));
}
