import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import WaveAnimation from '@site/src/components/WaveAnimation';

import styles from './index.module.css';

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
            <p className={styles.versionText}>Currently v0.5</p>
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
        <div className={styles.cardHeader}>Qumat 0.5 Released!</div>
        <div className={styles.cardBody}>
          <p>Mahout's new quantum computing layer for building ML circuits on simulators and real quantum hardware.</p>
          <Link to="/docs/download">Download Qumat 0.5 →</Link>
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
        computing that provides:
      </p>
      <ul>
        <li>
          <strong>Qumat Core</strong> - Build quantum circuits with standard
          gates and run them on Qiskit, Cirq, or Amazon Braket with a single
          unified API
        </li>
        <li>
          <strong>QDP (Quantum Data Plane)</strong> - Encode classical data
          into quantum states using GPU-accelerated kernels with zero-copy
          tensor transfer
        </li>
      </ul>
      <p>
        <Link to="/docs/qumat">Learn more about Qumat →</Link>
      </p>
    </div>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="Apache Mahout - Distributed Linear Algebra and Quantum Computing for Machine Learning">
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
