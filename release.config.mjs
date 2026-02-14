/**
 * Root semantic-release configuration.
 *
 * Generates GitHub Releases with changelogs for the monorepo root.
 * Python package (phspectra) has its own release pipeline via
 * python-semantic-release — see packages/phspectra/pyproject.toml.
 */
export default {
  branches: ['main'],
  tagFormat: 'v${version}',
  plugins: [
    '@semantic-release/commit-analyzer',
    '@semantic-release/release-notes-generator',
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md'
      }
    ],
    [
      '@semantic-release/npm',
      {
        npmPublish: false
      }
    ],
    [
      '@semantic-release/git',
      {
        assets: ['CHANGELOG.md', 'package.json'],
        message: 'chore(release): ${nextRelease.version} [skip ci]'
      }
    ],
    '@semantic-release/github'
  ]
}
