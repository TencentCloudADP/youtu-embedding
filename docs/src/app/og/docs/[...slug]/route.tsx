import { source } from '@/lib/source';
import { createImageResponse } from 'fumadocs-ui/og';
import { notFound } from 'next/navigation';

export const GET = createImageResponse({
  width: 1200,
  height: 630,
  async getImageResponse(ctx, { params }) {
    const page = source.getPage(params.slug);
    if (!page) notFound();

    return ctx.res({
      title: page.data.title,
      description: page.data.description,
    });
  },
});

export function generateStaticParams() {
  return source.generateParams();
}