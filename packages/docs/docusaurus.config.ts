import fs from 'node:fs'
import path from 'node:path'

import type * as Preset from '@docusaurus/preset-classic'
import type { Config } from '@docusaurus/types'
import { themes as prismThemes } from 'prism-react-renderer'
import rehypeKatex from 'rehype-katex'
import remarkMath from 'remark-math'

const baseUrl = process.env.DOCS_BASE_URL ?? '/phspectra/'

const pyprojectPath = path.resolve(__dirname, '../phspectra/pyproject.toml')
const pyprojectContent = fs.readFileSync(pyprojectPath, 'utf-8')
const versionMatch = pyprojectContent.match(/^version\s*=\s*"([^"]+)"/m)
const phspectraVersion = versionMatch?.[1] ?? '0.0.0'

const rootPkgPath = path.resolve(__dirname, '../../package.json')
const rootPkg = JSON.parse(fs.readFileSync(rootPkgPath, 'utf-8'))
const projectVersion = rootPkg.version ?? '0.0.0'

const config: Config = {
  title: 'PHSpectra',
  tagline: 'Persistent homology for spectral line decomposition',
  favicon: 'img/favicon.svg',

  url: 'https://caverac.github.io',
  baseUrl,

  organizationName: 'cavera',
  projectName: 'phspectra',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  customFields: {
    phspectraVersion,
    projectVersion
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en']
  },

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+',
      crossorigin: 'anonymous'
    }
  ],

  markdown: {
    mermaid: true
  },

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        docsRouteBasePath: '/',
        indexBlog: false
      }
    ]
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/caverac/phspectra/tree/main/packages/docs/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex]
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css'
        }
      } satisfies Preset.Options
    ]
  ],

  themeConfig: {
    navbar: {
      title: 'PHSpectra',
      logo: {
        alt: 'PHSpectra logo',
        src: 'img/logo.svg'
      },
      items: [
        {
          type: 'custom-projectVersionBadge',
          position: 'left'
        } as never,
        {
          type: 'custom-libraryVersionBadge',
          position: 'left'
        } as never,
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation'
        },
        {
          href: 'https://github.com/caverac/phspectra',
          label: 'GitHub',
          position: 'right'
        }
      ]
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: '/'
            },
            {
              label: 'Motivation',
              to: '/idea-and-plan/motivation'
            }
          ]
        }
      ],
      copyright: `Copyright \u00a9 ${new Date().getFullYear()} PHSpectra. Built with Docusaurus.`
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml']
    },
    mermaid: {
      theme: { light: 'neutral', dark: 'dark' }
    }
  } satisfies Preset.ThemeConfig
}

export default config
