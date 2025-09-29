import * as react_jsx_runtime from 'react/jsx-runtime';
import { ComponentProps, FC, ReactNode } from 'react';
import { StaticImport } from 'next/dist/shared/lib/get-img-props';

interface ImageProps extends Omit<ComponentProps<'img'>, 'src'> {
    sizes?: string;
    /**
     * Next.js Image component has other allowed type for `src`
     */
    src?: string | StaticImport;
    /**
     * priority of image (from Next.js)
     */
    priority?: boolean;
}
interface LinkProps extends ComponentProps<'a'> {
    prefetch?: boolean;
}
interface Router {
    push: (url: string) => void;
    refresh: () => void;
}
interface Framework {
    usePathname: () => string;
    useParams: () => Record<string, string | string[]>;
    useRouter: () => Router;
    Link?: FC<ComponentProps<'a'> & {
        prefetch?: boolean;
    }>;
    Image?: FC<ImageProps>;
}
declare function FrameworkProvider({ Link, useRouter, useParams, usePathname, Image, children, }: Framework & {
    children: ReactNode;
}): react_jsx_runtime.JSX.Element;
declare function usePathname(): string;
declare function useRouter(): Router;
declare function useParams(): Record<string, string | string[]>;
declare function Image(props: ImageProps): react_jsx_runtime.JSX.Element;
declare function Link(props: LinkProps): react_jsx_runtime.JSX.Element;
declare function createContext<T>(name: string, v?: T): {
    Provider: (props: {
        value: T;
        children: ReactNode;
    }) => react_jsx_runtime.JSX.Element;
    use: (errorMessage?: string) => Exclude<T, undefined | null>;
};

export { type Framework, FrameworkProvider, Image, type ImageProps, Link, type Router, createContext, useParams, usePathname, useRouter };
