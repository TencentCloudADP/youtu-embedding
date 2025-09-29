import { BunPlugin } from 'bun';

interface MdxPluginOptions {
    configPath?: string;
}
declare function createMdxPlugin(options?: MdxPluginOptions): BunPlugin;

export { type MdxPluginOptions, createMdxPlugin };
