import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import type { ReactElement } from 'react'

export function ProjectVersionBadge(): ReactElement {
  const { siteConfig } = useDocusaurusContext()
  const version = siteConfig.customFields?.projectVersion as string

  return <span className="version-badge">project v{version}</span>
}

export function LibraryVersionBadge(): ReactElement {
  const { siteConfig } = useDocusaurusContext()
  const version = siteConfig.customFields?.phspectraVersion as string

  return <span className="version-badge">phspectra v{version}</span>
}
