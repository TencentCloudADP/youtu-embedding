import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

/**
 * Shared layout configurations
 *
 * you can customise layouts individually from:
 * Home Layout: app/(home)/layout.tsx
 * Docs Layout: app/docs/layout.tsx
 */
export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <>
          <img 
            src="/logo.svg" 
            alt="Logo" 
            width="24" 
            height="24"
            className="rounded"
          />
          Youtu-Embedding
        </>
      ),
    },
    links: [],
  };
}

