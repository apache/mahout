import type {ReactElement} from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

export default function DownloadsPage(): ReactElement {
  return (
    <Layout title="Downloads" description="Apache Mahout download and install instructions">
      <main className="container margin-vert--lg">
        <Heading as="h1">Downloads</Heading>
        <p>Install Qumat from PyPI:</p>
        <pre>
          <code>pip install qumat</code>
        </pre>
        <p>Install with QDP (Quantum Data Plane) support:</p>
        <pre>
          <code>pip install qumat[qdp]</code>
        </pre>

        <Heading as="h2">From Source</Heading>
        <pre>
          <code>
            git clone https://github.com/apache/mahout.git{'\n'}
            cd mahout{'\n'}
            pip install uv{'\n'}
            uv sync                     # Core Qumat{'\n'}
            uv sync --extra qdp         # With QDP (requires CUDA GPU)
          </code>
        </pre>

        <Heading as="h2">Apache Release</Heading>
        <p>
          Official source releases are available at{' '}
          <a href="http://www.apache.org/dist/mahout">apache.org/dist/mahout</a>.
        </p>
        <p>To verify the integrity of a downloaded release:</p>
        <pre>
          <code>
            gpg --import KEYS{'\n'}
            gpg --verify mahout-qumat-0.5.zip.asc mahout-qumat-0.5.zip
          </code>
        </pre>

        <Heading as="h2">Links</Heading>
        <ul>
          <li>
            <a href="https://pypi.org/project/qumat/">PyPI</a>
          </li>
          <li>
            <a href="http://www.apache.org/dist/mahout">Apache SVN</a>
          </li>
        </ul>
      </main>
    </Layout>
  );
}
