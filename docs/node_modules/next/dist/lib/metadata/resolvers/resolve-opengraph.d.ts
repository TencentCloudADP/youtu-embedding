import type { ResolvedMetadata } from '../types/metadata-interface';
import type { OpenGraph } from '../types/opengraph-types';
import type { FieldResolverExtraArgs, AsyncFieldResolverExtraArgs, MetadataContext } from '../types/resolvers';
import type { Twitter } from '../types/twitter-types';
type ResolvedMetadataBase = ResolvedMetadata['metadataBase'];
export declare function resolveImages(images: Twitter['images'], metadataBase: ResolvedMetadataBase, isStaticMetadataRouteFile: boolean): NonNullable<ResolvedMetadata['twitter']>['images'];
export declare function resolveImages(images: OpenGraph['images'], metadataBase: ResolvedMetadataBase, isStaticMetadataRouteFile: boolean): NonNullable<ResolvedMetadata['openGraph']>['images'];
export declare const resolveOpenGraph: AsyncFieldResolverExtraArgs<'openGraph', [
    ResolvedMetadataBase,
    Promise<string>,
    MetadataContext,
    string | null
]>;
export declare const resolveTwitter: FieldResolverExtraArgs<'twitter', [
    ResolvedMetadataBase,
    MetadataContext,
    string | null
]>;
export {};
