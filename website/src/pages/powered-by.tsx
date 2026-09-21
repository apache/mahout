import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import entries from '../data/powered-by.json';
import styles from './powered-by.module.css';

type Entry = {
  name: string;
  url: string;
  category?: string;
  description?: string;
  source?: string;
  logo?: string;
};

const SUBMIT_URL =
  'https://github.com/apache/mahout/issues/new?template=powered-by.yml';
const DATA_FILE_URL =
  'https://github.com/apache/mahout/blob/main/website/src/data/powered-by.json';

const allEntries = entries as Entry[];
const featured = allEntries.filter((entry) => entry.description);
const everyone = [...allEntries].sort((a, b) =>
  a.name.localeCompare(b.name, 'en', {sensitivity: 'base'}),
);

function Logo({entry, className}: {entry: Entry; className: string}) {
  const src = useBaseUrl(entry.logo ?? '');
  if (!entry.logo) {
    return null;
  }
  return <img className={className} src={src} alt={`${entry.name} logo`} />;
}

function FeaturedCard({entry}: {entry: Entry}) {
  return (
    <article className={styles.card}>
      <div className={styles.cardHeader}>
        <div>
          <h3 className={styles.cardName}>
            <Link href={entry.url}>{entry.name}</Link>
          </h3>
          {entry.category && (
            <span className={styles.cardCategory}>{entry.category}</span>
          )}
        </div>
        {entry.logo && (
          <span className={styles.logoTile}>
            <Logo entry={entry} className={styles.cardLogo} />
          </span>
        )}
      </div>
      <p className={styles.cardBody}>{entry.description}</p>
      {entry.source && (
        <p className={styles.cardSource}>
          <Link href={entry.source}>Source</Link>
        </p>
      )}
    </article>
  );
}

function WallItem({entry}: {entry: Entry}) {
  return (
    <Link className={styles.wallItem} href={entry.url} title={entry.name}>
      {entry.logo ? (
        <Logo entry={entry} className={styles.wallLogo} />
      ) : (
        <span className={styles.wallName}>{entry.name}</span>
      )}
    </Link>
  );
}

export default function PoweredBy(): ReactNode {
  return (
    <Layout
      title="Powered By"
      description="Organizations, projects, and research groups using Apache Mahout.">
      <header className={styles.hero}>
        <div className="container">
          <div className={styles.heroInner}>
            <Heading as="h1" className={styles.title}>
              Powered By Apache Mahout
            </Heading>
            <p className={styles.lead}>
              Apache Mahout has powered recommendation, clustering, and
              classification systems at scale for more than a decade. Today the
              project builds Qumat, a Python library for writing quantum
              circuits once and running them on Qiskit, Cirq, or Amazon Braket,
              and QDP, a GPU-accelerated engine for encoding classical data into
              quantum states.
            </p>
            <p className={styles.lead}>
              See how organizations, developers, and researchers have built with
              Mahout, and add your own story.
            </p>
            <Link className="button button--primary button--lg" href={SUBMIT_URL}>
              Add your organization
            </Link>
          </div>
        </div>
      </header>

      <main>
        {featured.length > 0 && (
          <section className={styles.section}>
            <div className="container">
              <Heading as="h2" className={styles.sectionTitle}>
                How Mahout is being used
              </Heading>
              <p className={styles.sectionSubtitle}>
                Use cases the community has shared.
              </p>
              <div className={styles.cards}>
                {featured.map((entry) => (
                  <FeaturedCard key={entry.name} entry={entry} />
                ))}
              </div>
            </div>
          </section>
        )}

        <section className={styles.section}>
          <div className="container">
            <Heading as="h2" className={styles.sectionTitle}>
              Organizations building with Mahout
            </Heading>
            <p className={styles.sectionSubtitle}>
              Companies, projects, and research groups that have told the
              community they use or have used Apache Mahout.
            </p>
            <div className={styles.wall}>
              {everyone.map((entry) => (
                <WallItem key={entry.name} entry={entry} />
              ))}
            </div>
          </div>
        </section>

        <section className={styles.cta}>
          <div className="container">
            <div className={styles.ctaBox}>
              <Heading as="h2" className={styles.ctaTitle}>
                Using Mahout? Tell the community.
              </Heading>
              <p className={styles.ctaText}>
                Running Qumat in production, evaluating QDP, teaching a course,
                publishing research, or still on Mahout Classic? You are part of
                the story. Open a GitHub issue with the form below and a
                committer will add you to this page.
              </p>
              <div className={styles.ctaActions}>
                <Link className="button button--primary" href={SUBMIT_URL}>
                  Share your use case on GitHub
                </Link>
                <Link
                  className="button button--secondary"
                  to="/docs/community/mailing-lists">
                  Or email the dev list
                </Link>
              </div>
              <p className={styles.ctaAlt}>
                Prefer to send the pull request yourself? Add an entry to{' '}
                <Link href={DATA_FILE_URL}>powered-by.json</Link> in the
                website source.
              </p>
            </div>

            <p className={styles.disclaimer}>
              <strong>Disclaimer.</strong> Usage information on this page comes
              from community submissions and publicly available sources such as
              blog posts, conference talks, and the previous Mahout website.
              Usage, deployment stage, and other details may change over time.
              All company names, product names, trademarks, and logos shown here
              are the property of their respective owners and are used for
              identification purposes only. Inclusion on this page does not
              imply endorsement of Apache Mahout by these organizations, nor
              endorsement of these organizations by the Apache Software
              Foundation. If you represent an organization listed here and would
              like your entry or logo updated or removed, please{' '}
              <Link href="https://github.com/apache/mahout/issues/new?template=powered-by.yml">
                open an issue
              </Link>{' '}
              or email dev@mahout.apache.org.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
