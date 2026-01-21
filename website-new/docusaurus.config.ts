import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Apache Mahout',
  tagline: 'Distributed Linear Algebra & Quantum Computing',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://mahout.apache.org',
  baseUrl: '/',

  organizationName: 'apache',
  projectName: 'mahout',

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Use standard markdown format to avoid MDX parsing issues with HTML-like content
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/apache/mahout/tree/main/docs/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          // Versioning configuration - version 0.4 snapshot will be created after restructure
          lastVersion: 'current',
          versions: {
            current: {
              label: '0.5-dev',
              path: '',
            },
          },
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/apache/mahout/tree/main/website-new/',
          blogTitle: 'News & Updates',
          blogDescription: 'Apache Mahout project news, releases, and community updates',
          postsPerPage: 10,
          blogSidebarCount: 'ALL',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          onUntruncatedBlogPosts: 'ignore',
          onInlineTags: 'ignore',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  // KaTeX stylesheet for math rendering
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        // Only include redirects for paths that are known to exist
        // Additional redirects can be added after build verification
        createRedirects(existingPath) {
          // Redirect /news.html to /blog
          if (existingPath === '/blog') {
            return ['/news.html', '/news'];
          }
          return undefined;
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/mahout-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Apache Mahout',
      logo: {
        alt: 'Apache Mahout Logo',
        src: 'img/mahout-logo-blue.svg',
      },
      items: [
        // About dropdown
        {
          type: 'dropdown',
          label: 'About',
          position: 'left',
          items: [
            {
              label: 'Overview of the Apache Software Foundation',
              href: 'https://www.apache.org/foundation/how-it-works.html',
            },
            {
              label: 'How to Contribute',
              to: '/docs/about/how-to-contribute',
            },
          ],
        },
        // Qumat dropdown
        {
          type: 'dropdown',
          label: 'Qumat',
          position: 'left',
          items: [
            {
              label: 'Overview',
              to: '/docs/qumat',
            },
            {
              label: 'Qumat Core',
              to: '/docs/qumat/core',
            },
            {
              label: 'QDP (Quantum Data Plane)',
              to: '/docs/qumat/qdp',
            },
            {
              label: 'Quantum Computing Primer',
              to: '/docs/qumat/quantum-computing-primer',
            },
            {
              label: 'Papers',
              to: '/docs/qumat/papers',
            },
          ],
        },
        // Download
        {
          to: '/docs/download',
          label: 'Download',
          position: 'left',
        },
        // Community dropdown
        {
          type: 'dropdown',
          label: 'Community',
          position: 'left',
          items: [
            {
              label: 'Overview',
              to: '/docs/community',
            },
            {
              label: 'Who We Are',
              to: '/docs/community/who-we-are',
            },
            {
              label: 'Mailing Lists',
              to: '/docs/community/mailing-lists',
            },
            {
              label: 'Issue Tracker',
              href: 'https://issues.apache.org/jira/browse/MAHOUT',
            },
            {
              label: 'Code of Conduct',
              to: '/docs/community/code-of-conduct',
            },
          ],
        },
        // News (Blog)
        {
          to: '/blog',
          label: 'News',
          position: 'left',
        },
        // Version dropdown
        {
          type: 'docsVersionDropdown',
          position: 'right',
        },
        // GitHub
        {
          href: 'https://github.com/apache/mahout',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Qumat Overview',
              to: '/docs/qumat',
            },
            {
              label: 'Getting Started',
              to: '/docs/getting-started',
            },
            {
              label: 'API Reference',
              to: '/docs/api',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Mailing Lists',
              to: '/docs/community/mailing-lists',
            },
            {
              label: 'Issue Tracker',
              href: 'https://issues.apache.org/jira/browse/MAHOUT',
            },
            {
              label: 'Who We Are',
              to: '/docs/community/who-we-are',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'News',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/apache/mahout',
            },
            {
              label: 'Apache Software Foundation',
              href: 'https://www.apache.org/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} The Apache Software Foundation. Apache Mahout, Mahout, Apache, the Apache feather logo, and the Apache Mahout project logo are either registered trademarks or trademarks of The Apache Software Foundation in the United States and other countries.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'java', 'scala', 'rust', 'bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
