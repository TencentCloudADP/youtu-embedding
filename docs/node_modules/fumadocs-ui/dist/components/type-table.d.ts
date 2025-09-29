import { type ReactNode } from 'react';
export interface ParameterNode {
    name: string;
    description: ReactNode;
}
export interface TypeNode {
    /**
     * Additional description of the field
     */
    description?: ReactNode;
    /**
     * type signature (short)
     */
    type: ReactNode;
    /**
     * type signature (full)
     */
    typeDescription?: ReactNode;
    /**
     * Optional `href` for the type
     */
    typeDescriptionLink?: string;
    default?: ReactNode;
    required?: boolean;
    deprecated?: boolean;
    parameters?: ParameterNode[];
    returns?: ReactNode;
}
export declare function TypeTable({ type }: {
    type: Record<string, TypeNode>;
}): import("react/jsx-runtime").JSX.Element;
//# sourceMappingURL=type-table.d.ts.map