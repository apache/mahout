import React, {type ReactNode} from 'react';
import clsx from 'clsx';

import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {
  PageMetadata,
  HtmlClassNameProvider,
  ThemeClassNames,
} from '@docusaurus/theme-common';
import {useDateTimeFormat} from '@docusaurus/theme-common/internal';
import Layout from '@theme/Layout';
import BlogListPaginator from '@theme/BlogListPaginator';
import SearchMetadata from '@theme/SearchMetadata';
import type {Props} from '@theme/BlogListPage';
import BlogListPageStructuredData from '@theme/BlogListPage/StructuredData';

import styles from './styles.module.css';

type BlogListItem = Props['items'][number];

function useFormattedDate() {
  const dateTimeFormat = useDateTimeFormat({
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });

  return (blogDate: string) => dateTimeFormat.format(new Date(blogDate));
}

function ReadingTime({readingTime}: {readingTime?: number}) {
  if (typeof readingTime === 'undefined') {
    return null;
  }

  return <span>{Math.ceil(readingTime)} min read</span>;
}

function PostEyebrow({item}: {item: BlogListItem}) {
  const {date, readingTime} = item.content.metadata;
  const formatDate = useFormattedDate();

  return (
    <div className={styles.eyebrow}>
      <time dateTime={date}>{formatDate(date)}</time>
      <ReadingTime readingTime={readingTime} />
    </div>
  );
}

function Category({item}: {item: BlogListItem}) {
  const label = item.content.metadata.tags[0]?.label ?? 'News';

  return <div className={styles.category}>{label}</div>;
}

function Authors({item}: {item: BlogListItem}) {
  const names = item.content.metadata.authors
    .map((author) => author.name)
    .filter(Boolean);

  if (names.length === 0) {
    return null;
  }

  return <div className={styles.authors}>{names.join(', ')}</div>;
}

function isMeetingMinutes(item: BlogListItem) {
  const {title, tags} = item.content.metadata;

  return (
    title.toLowerCase().includes('meeting minutes') ||
    tags.some((tag) => tag.label.toLowerCase() === 'minutes')
  );
}

function PostCard({item}: {item: BlogListItem}) {
  const {metadata} = item.content;

  return (
    <article className={styles.card}>
      <Category item={item} />
      <PostEyebrow item={item} />
      <h2 className={styles.cardTitle}>
        <Link to={metadata.permalink}>{metadata.title}</Link>
      </h2>
      {metadata.description && (
        <p className={styles.cardDescription}>{metadata.description}</p>
      )}
      <div className={styles.cardFooter}>
        <Authors item={item} />
        <Link className={styles.readMore} to={metadata.permalink}>
          Read more
        </Link>
      </div>
    </article>
  );
}

function MinutesItem({item}: {item: BlogListItem}) {
  const {metadata} = item.content;

  return (
    <article className={styles.minutesItem}>
      <PostEyebrow item={item} />
      <h3 className={styles.minutesTitle}>
        <Link to={metadata.permalink}>{metadata.title}</Link>
      </h3>
      {metadata.description && (
        <p className={styles.minutesDescription}>{metadata.description}</p>
      )}
    </article>
  );
}

function FeaturedPost({item}: {item: BlogListItem}) {
  const {metadata} = item.content;

  return (
    <article className={styles.featured}>
      <div className={styles.featuredContent}>
        <Category item={item} />
        <PostEyebrow item={item} />
        <h2 className={styles.featuredTitle}>
          <Link to={metadata.permalink}>{metadata.title}</Link>
        </h2>
        {metadata.description && (
          <p className={styles.featuredDescription}>{metadata.description}</p>
        )}
        <div className={styles.featuredFooter}>
          <Authors item={item} />
          <Link className={styles.featuredLink} to={metadata.permalink}>
            Read the latest
          </Link>
        </div>
      </div>
      <div className={styles.featuredMeta} aria-hidden="true">
        <span>Latest</span>
        <strong>{metadata.tags[0]?.label ?? 'News'}</strong>
      </div>
    </article>
  );
}

function BlogListPageMetadata(props: Props): ReactNode {
  const {metadata} = props;
  const {
    siteConfig: {title: siteTitle},
  } = useDocusaurusContext();
  const {blogDescription, blogTitle, permalink} = metadata;
  const isBlogOnlyMode = permalink === '/';
  const title = isBlogOnlyMode ? siteTitle : blogTitle;
  return (
    <>
      <PageMetadata title={title} description={blogDescription} />
      <SearchMetadata tag="blog_posts_list" />
    </>
  );
}

function BlogListHeader({metadata}: Pick<Props, 'metadata'>) {
  return (
    <header className={styles.header}>
      <div>
        <div className={styles.kicker}>Apache Mahout</div>
        <h1 className={styles.title}>News</h1>
      </div>
      <p className={styles.description}>{metadata.blogDescription}</p>
    </header>
  );
}

function BlogListPageContent(props: Props): ReactNode {
  const {metadata, items} = props;
  const posts = items.filter((item) => !isMeetingMinutes(item));
  const minutes = items.filter(isMeetingMinutes);
  const [featured, ...blogPosts] = posts;

  return (
    <Layout>
      <main className="container margin-vert--lg">
        <BlogListHeader metadata={metadata} />
        <nav className={styles.sectionNav} aria-label="News sections">
          <a href="#blog">Blog</a>
          <a href="#meeting-minutes">Meeting Minutes</a>
        </nav>
        <section id="blog" className={styles.section}>
          <div className={styles.sectionHeader}>
            <span>Blog</span>
            <h2>Project updates and release notes</h2>
          </div>
        {featured && <FeaturedPost item={featured} />}
        <section className={styles.grid} aria-label="News posts">
          {blogPosts.map((item) => (
            <PostCard
              key={item.content.metadata.permalink}
              item={item}
            />
          ))}
        </section>
        </section>
        {minutes.length > 0 && (
          <section id="meeting-minutes" className={styles.minutesSection}>
            <div className={styles.sectionHeader}>
              <span>Meeting Minutes</span>
              <h2>Community meeting notes</h2>
            </div>
            <div className={styles.minutesList}>
              {minutes.map((item) => (
                <MinutesItem
                  key={item.content.metadata.permalink}
                  item={item}
                />
              ))}
            </div>
          </section>
        )}
        <BlogListPaginator metadata={metadata} />
      </main>
    </Layout>
  );
}

export default function BlogListPage(props: Props): ReactNode {
  return (
    <HtmlClassNameProvider
      className={clsx(
        ThemeClassNames.wrapper.blogPages,
        ThemeClassNames.page.blogListPage,
      )}>
      <BlogListPageMetadata {...props} />
      <BlogListPageStructuredData {...props} />
      <BlogListPageContent {...props} />
    </HtmlClassNameProvider>
  );
}
