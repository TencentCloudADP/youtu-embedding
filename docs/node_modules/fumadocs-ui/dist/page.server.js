import { jsx as _jsx } from "react/jsx-runtime";
import { getPageTreePeers } from 'fumadocs-core/server';
import { Card, Cards } from './components/card.js';
import * as path from 'node:path';
/**
 * @deprecated use https://fumadocs.vercel.app/docs/ui/markdown#further-reading-section instead
 */
export function DocsCategory({ page, from, tree: forcedTree, ...props }) {
    let tree;
    if (forcedTree) {
        tree = forcedTree;
    }
    else if (from._i18n) {
        const locale = page.locale ?? from._i18n.defaultLanguage;
        tree = from.pageTree[locale];
    }
    else {
        tree = from.pageTree;
    }
    let items = getPageTreePeers(tree, page.url);
    if (items.length === 0) {
        const pages = from.getPages(page.locale);
        items = pages
            .filter((item) => path.dirname(item.path) === path.dirname(page.path) &&
            item.path !== page.path)
            .map((page) => ({
            type: 'page',
            name: page.data.title,
            description: page.data.description,
            url: page.url,
        }));
    }
    if (items.length === 0)
        return null;
    return (_jsx(Cards, { ...props, children: items.map((item) => (_jsx(Card, { title: item.name, description: item.description, href: item.url }, item.url))) }));
}
export * from './page.js';
