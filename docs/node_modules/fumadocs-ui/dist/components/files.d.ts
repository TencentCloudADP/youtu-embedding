import { type HTMLAttributes, type ReactNode } from 'react';
export declare function Files({ className, ...props }: HTMLAttributes<HTMLDivElement>): React.ReactElement;
export interface FileProps extends HTMLAttributes<HTMLDivElement> {
    name: string;
    icon?: ReactNode;
}
export interface FolderProps extends HTMLAttributes<HTMLDivElement> {
    name: string;
    disabled?: boolean;
    /**
     * Open folder by default
     *
     * @defaultValue false
     */
    defaultOpen?: boolean;
}
export declare function File({ name, icon, className, ...rest }: FileProps): React.ReactElement;
export declare function Folder({ name, defaultOpen, ...props }: FolderProps): React.ReactElement;
//# sourceMappingURL=files.d.ts.map