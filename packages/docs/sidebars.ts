import type { SidebarsConfig } from '@docusaurus/plugin-content-docs'

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Idea and Plan of Attack',
      items: [
        'idea-and-plan/motivation',
        'idea-and-plan/persistent-homology-primer',
        'idea-and-plan/plan-of-attack',
        'idea-and-plan/data-sources'
      ]
    },
    {
      type: 'category',
      label: 'Infrastructure',
      items: ['infrastructure/aws-pipeline']
    }
  ]
}

export default sidebars
