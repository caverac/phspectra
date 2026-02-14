import useDocusaurusContext from '@docusaurus/useDocusaurusContext'

export default function VersionBadge(): JSX.Element {
  const { siteConfig } = useDocusaurusContext()
  const version = siteConfig.customFields?.phspectraVersion as string

  return <span className="version-badge">v{version}</span>
}
