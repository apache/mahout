import type {ReactNode} from 'react';
import Head from '@docusaurus/Head';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import WaveAnimation from '@site/src/components/WaveAnimation';
import releasedVersions from '../../versions.json';

import styles from './index.module.css';

const latestReleasedVersion = releasedVersions[0] ?? 'next';
const siteUrl = 'https://mahout.apache.org';
const pageTitle = 'Apache Mahout';
const pageDescription =
  'Apache Mahout builds an environment for quickly creating scalable, performant machine learning applications. Its current focus is Qumat, a backend-agnostic quantum circuit library, and QDP, a GPU-accelerated quantum data encoding engine.';
const logoUrl = `${siteUrl}/img/mahout-favicon.png`;
const socialImageUrl = `${siteUrl}/img/mascot_with_text.png`;

const organizationJsonLd = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'Apache Mahout',
  url: siteUrl,
  logo: {
    '@type': 'ImageObject',
    url: logoUrl,
    width: 512,
    height: 512,
  },
  image: socialImageUrl,
  sameAs: [
    'https://github.com/apache/mahout',
    'https://www.apache.org/',
  ],
};

const websiteJsonLd = {
  '@context': 'https://schema.org',
  '@type': 'WebSite',
  name: 'Apache Mahout',
  url: siteUrl,
  inLanguage: 'en',
  publisher: {
    '@type': 'Organization',
    name: 'Apache Mahout',
    url: siteUrl,
    logo: {
      '@type': 'ImageObject',
      url: logoUrl,
      width: 512,
      height: 512,
    },
  },
};

function HomepageHeader() {
  return (
    <header className={styles.heroBanner}>
      <WaveAnimation />
      <div className={styles.heroContent}>
        <div className={styles.heroUpper}>
          <div className="container">
            <div className={styles.heroLogos}>
              <img
                src="/img/mahout-logo-white.svg"
                alt="Apache Mahout"
                className={styles.mahoutLogo}
              />
              <img
                src="/img/asf_new_logo.svg"
                alt="Apache Software Foundation"
                className={styles.asfLogo}
              />
            </div>
          </div>
        </div>
        <div className={styles.heroLower}>
          <div className="container">
            <Heading as="h2" className={styles.heroSubtitle}>
              For Creating Scalable Performant Machine Learning Applications
            </Heading>
            <div className={styles.buttons}>
              <a
                href="https://pypi.org/project/qumat/"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.pipInstall}>
                pip install qumat
              </a>
            </div>
            <p className={styles.versionText}>Currently v{latestReleasedVersion}</p>
          </div>
        </div>
      </div>
    </header>
  );
}

function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      {/* Qumat Release Banner */}
      <div className={styles.sidebarCard}>
        <div className={styles.cardHeader}>Qumat {latestReleasedVersion} Released!</div>
        <div className={styles.cardBody}>
          <p>
            First-class AMD ROCm support, six QDP encodings with parity across
            CUDA and ROCm, and faster zero-copy GPU paths.
          </p>
          <p>
            <Link to="/blog/2026/06/01/Qumat-0.6.0-Release">Read the release notes →</Link>
          </p>
          <Link to="/docs/qumat/getting-started">Get Qumat {latestReleasedVersion} →</Link>
        </div>
      </div>

      {/* Apache Software Foundation Links */}
      <div className={styles.sidebarCard}>
        <div className={styles.cardHeader}>Apache Software Foundation</div>
        <div className={styles.cardBody}>
          <ul className={styles.linkList}>
            <li>
              <a href="https://www.apache.org/foundation/how-it-works.html">
                Apache Software Foundation
              </a>
            </li>
            <li>
              <a href="http://www.apache.org/licenses/">Apache License</a>
            </li>
            <li>
              <a href="http://www.apache.org/foundation/sponsorship.html">
                Sponsorship
              </a>
            </li>
            <li>
              <a href="http://www.apache.org/foundation/thanks.html">Thanks</a>
            </li>
          </ul>
        </div>
      </div>

      {/* Talks Widget */}
      <div className={styles.sidebarCard}>
        <div className={styles.cardHeader}>Talks</div>
        <div className={styles.cardBody}>
          <ul className={styles.linkList}>
            <li>
              Community Over Code Asia 2026 -{' '}
              <a
                href="https://asia.communityovercode.org/sessions/general-1194670.html"
                target="_blank"
                rel="noopener noreferrer">
                Accelerating Quantum Machine Learning: Building a
                GPU-Accelerated Data Plane in Apache Mahout
              </a>
            </li>
            <li>
              FOSDEM 2025 -{' '}
              <a
                href="https://mirrors.dotsrc.org/fosdem/2025/k4401/fosdem-2025-5298-introducing-qumat-an-apache-mahout-joint-.av1.webm"
                target="_blank"
                rel="noopener noreferrer">
                Introducing Qumat!
              </a>
            </li>
            <li>
              FOSSY 2024 -{' '}
              <a
                href="https://www.youtube.com/watch?v=tgFaUL1wYhY"
                target="_blank"
                rel="noopener noreferrer">
                QuMat: Apache Mahout's Quantum Computing Interface
              </a>
            </li>
          </ul>
        </div>
      </div>
    </aside>
  );
}

function MainContent() {
  return (
    <div className={styles.mainContent}>
      <Heading as="h1">Apache Mahout</Heading>
      <p>
        The goal of the Apache Mahout™ project is to build an environment for
        quickly creating scalable, performant machine learning applications.
      </p>
      <p>
        Today that work centers on quantum machine learning in Python, with two
        components: <strong>Qumat</strong>, a library for writing quantum
        circuits once and running them on any supported backend, and{' '}
        <strong>QDP</strong>, a GPU-accelerated data plane that turns classical
        data into quantum states without simulating state-preparation
        circuits.
      </p>

      <Heading as="h2">Qumat</Heading>
      <div className={styles.mascotContainer}>
        <img
          src="/img/mascot_with_text.png"
          alt="Apache Mahout Qumat"
          className={styles.mascotImage}
        />
      </div>

      <p>
        <strong>Qumat</strong> is a high-level Python library for quantum
        computing. Build a circuit with standard and parameterized gates, then
        execute it on Qiskit, Cirq, or Amazon Braket through one unified API,
        on simulators or real quantum hardware.
      </p>
      <ul>
        <li>
          <strong>One API, three backends</strong> - Switch between Qiskit,
          Cirq, and Amazon Braket by changing a config value, not your circuit
          code
        </li>
        <li>
          <strong>Standard and parameterized gates</strong> - Hadamard, Pauli,
          CNOT, Toffoli, SWAP, and rotation gates with parameter binding for
          variational circuits
        </li>
        <li>
          <strong>Python 3.10+</strong> - Installable from PyPI with{' '}
          <code>pip install qumat</code>
        </li>
      </ul>
      <p>
        <Link to="/docs/qumat">Learn more about Qumat →</Link>
      </p>

      <Heading as="h2">QDP (Quantum Data Plane)</Heading>
      <p>
        <strong>QDP</strong> removes the data-loading bottleneck in quantum
        machine learning. Instead of simulating a state-preparation circuit,
        it constructs the state vector directly in GPU memory and hands it to
        your training or kernel pipeline.
      </p>
      <ul>
        <li>
          <strong>Six encodings</strong> - Amplitude, angle, basis, phase, IQP,
          and IQP-Z, with the same coverage on every GPU backend
        </li>
        <li>
          <strong>NVIDIA CUDA and AMD ROCm</strong> - Native CUDA kernels and
          hand-written Triton kernels for ROCm, selectable from the same API
        </li>
        <li>
          <strong>Zero-copy interop</strong> - DLPack handoff to and from
          PyTorch, NumPy, and TensorFlow, plus GPU-pointer paths that skip the
          host round trip
        </li>
        <li>
          <strong>Benchmarked on real workloads</strong> - SVHN IQP training,
          quantum kernel SVM, and data-to-state latency benchmarks ship with
          the project
        </li>
      </ul>
      <p>
        <Link to="/docs/qdp">Learn more about QDP →</Link>
      </p>

      <p className={styles.legacyNote}>
        Looking for the earlier Mahout Classic (Samsara and MapReduce)
        codebase? It is in maintenance mode. The community's current work is
        on Qumat and QDP.
      </p>
    </div>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description={pageDescription}>
      <Head>
        <meta property="og:title" content={pageTitle} />
        <meta property="og:description" content={pageDescription} />
        <meta property="og:type" content="website" />
        <meta property="og:url" content={siteUrl} />
        <meta property="og:image" content={socialImageUrl} />
        <meta property="og:image:alt" content="Apache Mahout Qumat mascot and logo" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={pageTitle} />
        <meta name="twitter:description" content={pageDescription} />
        <meta name="twitter:image" content={socialImageUrl} />
        <meta name="application-name" content={siteConfig.title} />
        <link rel="icon" href={logoUrl} sizes="512x512" type="image/png" />
        <link rel="apple-touch-icon" href={logoUrl} />
        <script type="application/ld+json">
          {JSON.stringify(organizationJsonLd)}
        </script>
        <script type="application/ld+json">
          {JSON.stringify(websiteJsonLd)}
        </script>
      </Head>
      <HomepageHeader />
      <main className={styles.mainSection}>
        <div className="container">
          <div className={styles.contentLayout}>
            <MainContent />
            <Sidebar />
          </div>
        </div>
      </main>
    </Layout>
  );
}
