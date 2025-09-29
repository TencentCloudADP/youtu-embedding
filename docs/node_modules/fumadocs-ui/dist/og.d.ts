import { ImageResponse } from 'next/og';
import type { ReactElement, ReactNode } from 'react';
import type { ImageResponseOptions } from 'next/dist/compiled/@vercel/og/types';
interface GenerateProps {
    title: ReactNode;
    description?: ReactNode;
    icon?: ReactNode;
    primaryColor?: string;
    primaryTextColor?: string;
    site?: ReactNode;
}
export declare function generateOGImage(options: GenerateProps & ImageResponseOptions): ImageResponse;
export declare function generate({ primaryColor, primaryTextColor, ...props }: GenerateProps): ReactElement;
export {};
//# sourceMappingURL=og.d.ts.map