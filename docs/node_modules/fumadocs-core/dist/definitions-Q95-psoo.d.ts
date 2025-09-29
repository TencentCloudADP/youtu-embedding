import { ReactNode } from 'react';

interface Root {
    $id?: string;
    name: ReactNode;
    children: Node[];
    /**
     * Another page tree that won't be displayed unless being opened.
     */
    fallback?: Root;
}
type Node = Item | Separator | Folder;
interface Item {
    $id?: string;
    /**
     * @internal
     */
    $ref?: {
        file: string;
    };
    type: 'page';
    name: ReactNode;
    url: string;
    external?: boolean;
    description?: ReactNode;
    icon?: ReactNode;
}
interface Separator {
    $id?: string;
    type: 'separator';
    name?: ReactNode;
    icon?: ReactNode;
}
interface Folder {
    $id?: string;
    /**
     * @internal
     */
    $ref?: {
        metaFile?: string;
    };
    type: 'folder';
    name: ReactNode;
    description?: ReactNode;
    root?: boolean;
    defaultOpen?: boolean;
    index?: Item;
    icon?: ReactNode;
    children: Node[];
}

type definitions_Folder = Folder;
type definitions_Item = Item;
type definitions_Node = Node;
type definitions_Root = Root;
type definitions_Separator = Separator;
declare namespace definitions {
  export type { definitions_Folder as Folder, definitions_Item as Item, definitions_Node as Node, definitions_Root as Root, definitions_Separator as Separator };
}

export { type Folder as F, type Item as I, type Node as N, type Root as R, type Separator as S, definitions as d };
