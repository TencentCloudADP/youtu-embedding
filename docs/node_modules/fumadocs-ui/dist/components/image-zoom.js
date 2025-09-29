'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { Image } from 'fumadocs-core/framework';
import './image-zoom.css';
import Zoom from 'react-medium-image-zoom';
function getImageSrc(src) {
    if (typeof src === 'string')
        return src;
    if (typeof src === 'object') {
        // Next.js
        if ('default' in src)
            return src.default.src;
        return src.src;
    }
    return '';
}
export function ImageZoom({ zoomInProps, children, rmiz, ...props }) {
    return (_jsx(Zoom, { zoomMargin: 20, wrapElement: "span", ...rmiz, zoomImg: {
            src: getImageSrc(props.src),
            sizes: undefined,
            ...zoomInProps,
        }, children: children ?? (_jsx(Image, { sizes: "(max-width: 768px) 100vw, (max-width: 1200px) 70vw, 900px", ...props })) }));
}
