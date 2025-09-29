import { type ImageProps } from 'fumadocs-core/framework';
import { type ImgHTMLAttributes } from 'react';
import './image-zoom.css';
import { type UncontrolledProps } from 'react-medium-image-zoom';
export type ImageZoomProps = ImageProps & {
    /**
     * Image props when zoom in
     */
    zoomInProps?: ImgHTMLAttributes<HTMLImageElement>;
    /**
     * Props for `react-medium-image-zoom`
     */
    rmiz?: UncontrolledProps;
};
export declare function ImageZoom({ zoomInProps, children, rmiz, ...props }: ImageZoomProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=image-zoom.d.ts.map