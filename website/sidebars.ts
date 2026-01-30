import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Apache Mahout Documentation Sidebars
 */
const sidebars: SidebarsConfig = {
  docsSidebar: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: ['qumat/getting-started'],
    },
    {
      type: 'category',
      label: 'Qumat (Circuits)',
      collapsed: false,
      link: {type: 'doc', id: 'qumat/index'},
      items: [
        'qumat/basic-gates',
        'qumat/parameterized-circuits',
        'qumat/api',
        'qumat/concepts',
        'qumat/examples',
      ],
    },
    {
      type: 'category',
      label: 'QDP (Data Encoding)',
      collapsed: false,
      link: {type: 'doc', id: 'qdp/index'},
      items: [
        'qdp/getting-started',
        'qdp/concepts',
        'qdp/api',
        'qdp/examples',
        {
          type: 'category',
          label: 'Internals',
          collapsed: true,
          items: [
            'qdp/readers',
            'qdp/observability',
            'qdp/testing',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Advanced Topics',
      collapsed: true,
      items: [
        'advanced/pqc',
        'advanced/gap-analysis',
      ],
    },
    {
      type: 'category',
      label: 'Quantum Computing Primer',
      collapsed: true,
      link: {type: 'doc', id: 'learning/quantum-computing-primer/index'},
      items: [
        'learning/quantum-computing-primer/introduction',
        'learning/quantum-computing-primer/qubits',
        'learning/quantum-computing-primer/quantum-gates',
        'learning/quantum-computing-primer/quantum-circuits',
        'learning/quantum-computing-primer/quantum-entanglement',
        'learning/quantum-computing-primer/quantum-algorithms',
        'learning/quantum-computing-primer/quantum-error-correction',
        'learning/quantum-computing-primer/applications',
        'learning/quantum-computing-primer/advanced-topics',
      ],
    },
    {
      type: 'category',
      label: 'Research Papers',
      collapsed: true,
      link: {type: 'doc', id: 'learning/papers/index'},
      items: [
        'learning/papers/An-Efficient-Quantum-Factoring-Algorithm',
        'learning/papers/Quantum-Kernel-Estimation-With-Neutral-Atoms-For-Supervised-Classification',
        'learning/papers/Quantum-machine-learning-beyond-kernel-methods',
        'learning/papers/Unleashing-the-Potential-of-LLMs-for-Quantum-Computing',
      ],
    },
    {
      type: 'category',
      label: 'Download',
      collapsed: true,
      link: {type: 'doc', id: 'download/index'},
      items: ['download/quickstart'],
    },
    {
      type: 'category',
      label: 'Community',
      collapsed: true,
      link: {type: 'doc', id: 'community/index'},
      items: [
        'community/who-we-are',
        'community/mailing-lists',
        'community/code-of-conduct',
      ],
    },
    {
      type: 'category',
      label: 'Contributing',
      collapsed: true,
      items: ['about/how-to-contribute'],
    },
  ],
};

export default sidebars;
