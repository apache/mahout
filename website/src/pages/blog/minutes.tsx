import React, {type ReactNode} from 'react';
import clsx from 'clsx';

import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import {
  HtmlClassNameProvider,
  PageMetadata,
  ThemeClassNames,
} from '@docusaurus/theme-common';
import Layout from '@theme/Layout';
import SearchMetadata from '@theme/SearchMetadata';
import blogPostList from '~blog/default/blog-post-list-prop-default.json';

import styles from '../../theme/BlogListPage/styles.module.css';

type BlogListEntry = (typeof blogPostList.items)[number];

const MINUTES_PER_PAGE = 12;

function getCurrentPage(search: string) {
  const rawPage = Number(new URLSearchParams(search).get('page'));

  return Number.isInteger(rawPage) && rawPage > 0 ? rawPage : 1;
}

function getPageLink(page: number) {
  return page > 1 ? `/blog/minutes?page=${page}` : '/blog/minutes';
}

function paginateItems(items: BlogListEntry[], page: number) {
  const totalPages = Math.max(1, Math.ceil(items.length / MINUTES_PER_PAGE));
  const currentPage = Math.min(page, totalPages);
  const start = (currentPage - 1) * MINUTES_PER_PAGE;

  return {
    currentPage,
    pageItems: items.slice(start, start + MINUTES_PER_PAGE),
    totalPages,
  };
}

function formatDate(date: string) {
  return new Intl.DateTimeFormat('en', {
    day: 'numeric',
    month: 'long',
    timeZone: 'UTC',
    year: 'numeric',
  }).format(new Date(date));
}

function isMeetingMinutes(item: BlogListEntry) {
  return item.title.toLowerCase().includes('meeting minutes');
}

function NewsHeader() {
  return (
    <header className={styles.header}>
      <div>
        <div className={styles.kicker}>Apache Mahout</div>
        <h1 className={styles.title}>News</h1>
      </div>
      <p className={styles.description}>
        Apache Mahout project news, releases, and community updates
      </p>
    </header>
  );
}

function NewsNav() {
  return (
    <nav className={styles.sectionNav} aria-label="News sections">
      <Link to="/blog">Blog</Link>
      <Link className={styles.activeNavLink} to="/blog/minutes">
        Meeting Minutes
      </Link>
    </nav>
  );
}

function MinutesItem({item}: {item: BlogListEntry}) {
  return (
    <article className={styles.minutesItem}>
      <div className={styles.eyebrow}>
        <time dateTime={item.date}>{formatDate(item.date)}</time>
      </div>
      <h3 className={styles.minutesTitle}>
        <Link to={item.permalink}>{item.title}</Link>
      </h3>
    </article>
  );
}

function ListPagination({
  currentPage,
  totalPages,
}: {
  currentPage: number;
  totalPages: number;
}) {
  if (totalPages <= 1) {
    return null;
  }

  const pages = Array.from({length: totalPages}, (_, index) => index + 1);

  return (
    <nav className={styles.pagination} aria-label="Meeting minutes pagination">
      <Link
        className={clsx(
          styles.pageButton,
          currentPage === 1 && styles.disabledPageButton,
        )}
        aria-disabled={currentPage === 1}
        tabIndex={currentPage === 1 ? -1 : undefined}
        to={getPageLink(Math.max(1, currentPage - 1))}>
        Previous
      </Link>
      <div className={styles.pageNumbers}>
        {pages.map((page) => (
          <Link
            key={page}
            className={clsx(
              styles.pageNumber,
              page === currentPage && styles.activePageNumber,
            )}
            aria-current={page === currentPage ? 'page' : undefined}
            to={getPageLink(page)}>
            {page}
          </Link>
        ))}
      </div>
      <Link
        className={clsx(
          styles.pageButton,
          currentPage === totalPages && styles.disabledPageButton,
        )}
        aria-disabled={currentPage === totalPages}
        tabIndex={currentPage === totalPages ? -1 : undefined}
        to={getPageLink(Math.min(totalPages, currentPage + 1))}>
        Next
      </Link>
    </nav>
  );
}

function MeetingMinutesPageContent() {
  const location = useLocation();
  const minutes = blogPostList.items.filter(isMeetingMinutes);
  const {currentPage, pageItems, totalPages} = paginateItems(
    minutes,
    getCurrentPage(location.search),
  );

  return (
    <Layout>
      <main className="container margin-vert--lg">
        <NewsHeader />
        <NewsNav />
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span>Meeting Minutes</span>
            <h2>Community meeting notes</h2>
          </div>
          <div className={styles.minutesList}>
            {pageItems.map((item) => (
              <MinutesItem key={item.permalink} item={item} />
            ))}
          </div>
        </section>
        <ListPagination currentPage={currentPage} totalPages={totalPages} />
      </main>
    </Layout>
  );
}

export default function MeetingMinutesPage(): ReactNode {
  return (
    <HtmlClassNameProvider
      className={clsx(
        ThemeClassNames.wrapper.blogPages,
        ThemeClassNames.page.blogListPage,
      )}>
      <PageMetadata
        title="Meeting Minutes"
        description="Apache Mahout community meeting notes"
      />
      <SearchMetadata tag="blog_posts_list" />
      <MeetingMinutesPageContent />
    </HtmlClassNameProvider>
  );
}
