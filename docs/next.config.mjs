import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  // GitHub Pages 配置
  basePath: process.env.NODE_ENV === 'production' ? '/Youtu-Embedding' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/Youtu-Embedding/' : '',
};

export default withMDX(config);
