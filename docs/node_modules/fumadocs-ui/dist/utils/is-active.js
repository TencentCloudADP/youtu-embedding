function normalize(url) {
    if (url.length > 1 && url.endsWith('/'))
        return url.slice(0, -1);
    return url;
}
export function isActive(url, pathname, nested = true) {
    url = normalize(url);
    pathname = normalize(pathname);
    return url === pathname || (nested && pathname.startsWith(`${url}/`));
}
export function isTabActive(tab, pathname) {
    if (tab.urls)
        return tab.urls.has(normalize(pathname));
    return isActive(tab.url, pathname, true);
}
