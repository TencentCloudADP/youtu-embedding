import { type HTMLAttributes } from 'react';
import { type BaseLayoutProps, type NavOptions } from '../../layouts/shared/index.js';
export interface HomeLayoutProps extends BaseLayoutProps {
    nav?: Partial<NavOptions & {
        /**
         * Open mobile menu when hovering the trigger
         */
        enableHoverToOpen?: boolean;
    }>;
}
export declare function HomeLayout(props: HomeLayoutProps & HTMLAttributes<HTMLElement>): import("react/jsx-runtime").JSX.Element;
export declare function Header({ nav, i18n, links, githubUrl, themeSwitch, searchToggle, }: HomeLayoutProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=index.d.ts.map