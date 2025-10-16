import './global.css';
import 'remixicon/fonts/remixicon.css';
import { RootProvider } from 'fumadocs-ui/provider';
import { Inter } from 'next/font/google';
import Script from 'next/script';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';

const inter = Inter({
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'ADP Chat Client - 学习和分享腾讯云智能体平台最佳实践',
  description: 'ADP-Chat-Client是一个开源的AI智能体应用对话端。可以将腾讯云智能体开发平台（Tencent Cloud ADP） 开发的AI智能体应用快速部署为Web应用（或嵌入到小程序、Android、iOS 应用中）。',
  icons: {
    icon: '/images/favicon.png',
    apple: '/images/favicon.png',
  },
  openGraph: {
    title: 'ADP Chat Client - 学习和分享腾讯云智能体平台最佳实践',
    description: 'ADP-Chat-Client是一个开源的AI智能体应用对话端。可以将腾讯云智能体开发平台（Tencent Cloud ADP） 开发的AI智能体应用快速部署为Web应用（或嵌入到小程序、Android、iOS 应用中）。',
    images: [
      {
        url: '/images/hello-adp.png',
        width: 1200,
        height: 630,
        alt: 'Hello ADP Logo',
      },
    ],
    locale: 'zh_CN',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Hello ADP - 学习和分享腾讯云智能体平台最佳实践',
    description: '帮助新手快速上手腾讯云智能体平台（Tencent Cloud Agent Development Platform，ADP）的教程',
    images: ['/images/hello-adp.png'],
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen" suppressHydrationWarning>
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-T8G1R1CX89"
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-T8G1R1CX89');
          `}
        </Script>
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
