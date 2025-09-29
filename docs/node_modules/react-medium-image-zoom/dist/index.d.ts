import React from 'react';

declare const enum ModalState {
    LOADED = "LOADED",
    LOADING = "LOADING",
    UNLOADED = "UNLOADED",
    UNLOADING = "UNLOADING"
}
interface ControlledProps {
    a11yNameButtonUnzoom?: string;
    a11yNameButtonZoom?: string;
    canSwipeToUnzoom?: boolean;
    children: React.ReactNode;
    classDialog?: string;
    IconUnzoom?: React.ElementType;
    IconZoom?: React.ElementType;
    isDisabled?: boolean;
    isZoomed: boolean;
    onZoomChange?: (value: boolean, data: {
        event: React.SyntheticEvent | Event;
    }) => void;
    swipeToUnzoomThreshold?: number;
    wrapElement?: 'div' | 'span';
    ZoomContent?: (data: {
        buttonUnzoom: React.ReactElement<HTMLButtonElement>;
        img: React.ReactElement | null;
        isZoomImgLoaded: boolean;
        modalState: ModalState;
        onUnzoom: (e: Event) => void;
    }) => React.ReactElement;
    zoomImg?: React.ImgHTMLAttributes<HTMLImageElement>;
    zoomMargin?: number;
}
declare function Controlled(props: ControlledProps): React.JSX.Element;

type UncontrolledProps = Omit<ControlledProps, 'isZoomed'>;
declare function Uncontrolled({ onZoomChange, ...props }: UncontrolledProps): React.JSX.Element;

export { Controlled, Uncontrolled as default };
export type { ControlledProps, UncontrolledProps };
