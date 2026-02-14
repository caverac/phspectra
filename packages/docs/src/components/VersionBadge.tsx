import type { ReactElement } from 'react'

import useDocusaurusContext from '@docusaurus/useDocusaurusContext'

export default function VersionBadge(): ReactElement {
  const { siteConfig } = useDocusaurusContext()
  const version = siteConfig.customFields?.phspectraVersion as string

  return <span className="version-badge">v{version}</span>
}
