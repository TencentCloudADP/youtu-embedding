import { type HTMLAttributes } from 'react';
type BannerVariant = 'rainbow' | 'normal';
export declare function Banner({ id, variant, changeLayout, height, rainbowColors, ...props }: HTMLAttributes<HTMLDivElement> & {
    /**
     * @defaultValue 3rem
     */
    height?: string;
    /**
     * @defaultValue 'normal'
     */
    variant?: BannerVariant;
    /**
     * For rainbow variant only, customise the colors
     */
    rainbowColors?: string[];
    /**
     * Change Fumadocs layout styles
     *
     * @defaultValue true
     */
    changeLayout?: boolean;
}): import("react/jsx-runtime").JSX.Element | null;
export {};
//# sourceMappingURL=banner.d.ts.map