import { type VariantProps } from 'class-variance-authority';
export declare const buttonVariants: (props?: ({
    variant?: "primary" | "outline" | "ghost" | "secondary" | null | undefined;
    color?: "primary" | "outline" | "ghost" | "secondary" | null | undefined;
    size?: "sm" | "icon" | "icon-sm" | "icon-xs" | null | undefined;
} & import("class-variance-authority/types").ClassProp) | undefined) => string;
export type ButtonProps = VariantProps<typeof buttonVariants>;
//# sourceMappingURL=button.d.ts.map