import type { SidebarsConfig } from '@docusaurus/plugin-content-docs'

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Preliminaries',
      items: [
        'idea-and-plan/motivation',
        'idea-and-plan/persistent-homology-primer',
        'idea-and-plan/data-sources'
      ]
    },
    {
      type: 'category',
      label: 'Results',
      items: ['results/beta', 'results/accuracy', 'results/performance', 'results/reproducing']
    },
    {
      type: 'category',
      label: 'Infrastructure',
      items: ['infrastructure/aws-pipeline']
    },
    {
      type: 'category',
      label: 'PHSpectra Library',
      items: ['library/installation', 'library/api', 'library/algorithm', 'library/examples']
    }
  ]
}

export default sidebars
