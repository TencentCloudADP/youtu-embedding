import { type LinkItemType } from '../../layouts/shared/index.js';
import { NavigationMenuContent, NavigationMenuTrigger } from '../../components/ui/navigation-menu.js';
import type { ComponentPropsWithoutRef } from 'react';
export declare function MenuLinkItem({ item, ...props }: {
    item: LinkItemType;
    className?: string;
}): import("react/jsx-runtime").JSX.Element;
export declare const Menu: import("react").ForwardRefExoticComponent<Omit<import("@radix-ui/react-navigation-menu").NavigationMenuItemProps & import("react").RefAttributes<HTMLLIElement>, "ref"> & import("react").RefAttributes<HTMLLIElement>>;
export declare function MenuTrigger({ enableHover, ...props }: ComponentPropsWithoutRef<typeof NavigationMenuTrigger> & {
    /**
     * Enable hover to trigger
     */
    enableHover?: boolean;
}): import("react/jsx-runtime").JSX.Element;
export declare function MenuContent(props: ComponentPropsWithoutRef<typeof NavigationMenuContent>): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=menu.d.ts.map