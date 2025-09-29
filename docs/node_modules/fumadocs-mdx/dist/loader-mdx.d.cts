import { LoaderContext } from 'webpack';

interface Options {
    configPath: string;
    outDir: string;
}
/**
 * Load MDX/markdown files
 *
 * it supports frontmatter by parsing and injecting the data in `vfile.data.frontmatter`
 */
declare function loader(this: LoaderContext<Options>, source: string, callback: LoaderContext<Options>['callback']): Promise<void>;

export { type Options, loader as default };
