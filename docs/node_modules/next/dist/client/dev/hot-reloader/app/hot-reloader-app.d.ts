import type { ReactNode } from 'react';
import type { GlobalErrorState } from '../../../components/app-router-instance';
export declare function waitForWebpackRuntimeHotUpdate(): Promise<void>;
export default function HotReload({ assetPrefix, children, globalError, }: {
    assetPrefix: string;
    children: ReactNode;
    globalError: GlobalErrorState;
}): import("react/jsx-runtime").JSX.Element;
