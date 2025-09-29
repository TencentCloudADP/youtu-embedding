import type { HTMLAttributes, ReactNode } from 'react';
export declare function Cards(props: HTMLAttributes<HTMLDivElement>): import("react/jsx-runtime").JSX.Element;
export type CardProps = Omit<HTMLAttributes<HTMLElement>, 'title'> & {
    icon?: ReactNode;
    title: ReactNode;
    description?: ReactNode;
    href?: string;
    external?: boolean;
};
export declare function Card({ icon, title, description, ...props }: CardProps): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=card.d.ts.map