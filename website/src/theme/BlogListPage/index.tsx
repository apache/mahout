import React, {type ReactNode} from 'react';
import clsx from 'clsx';

import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useLocation} from '@docusaurus/router';
import {
  PageMetadata,
  HtmlClassNameProvider,
  ThemeClassNames,
} from '@docusaurus/theme-common';
import {useDateTimeFormat} from '@docusaurus/theme-common/internal';
import Layout from '@theme/Layout';
import SearchMetadata from '@theme/SearchMetadata';
import type {Props} from '@theme/BlogListPage';
import BlogListPageStructuredData from '@theme/BlogListPage/StructuredData';

import styles from './styles.module.css';

type BlogListItem = Props['items'][number];
type NewsView = 'blog' | 'minutes';

const BLOG_POSTS_PER_PAGE = 9;

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

function getView(search: string): NewsView {
  const params = new URLSearchParams(search);

  return params.get('view') === 'minutes' ? 'minutes' : 'blog';
}

function getCurrentPage(search: string) {
  const rawPage = Number(new URLSearchParams(search).get('page'));

  return Number.isInteger(rawPage) && rawPage > 0 ? rawPage : 1;
}

function getPageLink(view: NewsView, page: number) {
  const params = new URLSearchParams();

  if (view === 'minutes') {
    if (page > 1) {
      params.set('page', String(page));
    }

    const query = params.toString();
    return query ? `/blog/minutes?${query}` : '/blog/minutes';
  }
  if (page > 1) {
    params.set('page', String(page));
  }

  const query = params.toString();
  return query ? `/blog?${query}` : '/blog';
}

function paginateItems(items: BlogListItem[], page: number, pageSize: number) {
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const currentPage = Math.min(page, totalPages);
  const start = (currentPage - 1) * pageSize;

  return {
    currentPage,
    pageItems: items.slice(start, start + pageSize),
    totalPages,
  };
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
        <h1 className={styles.title}>News</h1>
      </div>
      <p className={styles.description}>{metadata.blogDescription}</p>
    </header>
  );
}

function NewsNav({view}: {view: NewsView}) {
  return (
    <nav className={styles.sectionNav} aria-label="News sections">
      <Link
        className={clsx(view === 'blog' && styles.activeNavLink)}
        to="/blog">
        Blog
      </Link>
      <Link
        className={clsx(view === 'minutes' && styles.activeNavLink)}
        to="/blog/minutes">
        Meeting Minutes
      </Link>
    </nav>
  );
}

function ListPagination({
  currentPage,
  totalPages,
  view,
}: {
  currentPage: number;
  totalPages: number;
  view: NewsView;
}) {
  if (totalPages <= 1) {
    return null;
  }

  const pages = Array.from({length: totalPages}, (_, index) => index + 1);

  return (
    <nav className={styles.pagination} aria-label="News pagination">
      <Link
        className={clsx(
          styles.pageButton,
          currentPage === 1 && styles.disabledPageButton,
        )}
        aria-disabled={currentPage === 1}
        tabIndex={currentPage === 1 ? -1 : undefined}
        to={getPageLink(view, Math.max(1, currentPage - 1))}>
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
            to={getPageLink(view, page)}>
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
        to={getPageLink(view, Math.min(totalPages, currentPage + 1))}>
        Next
      </Link>
    </nav>
  );
}

function BlogListPageContent(props: Props): ReactNode {
  const {metadata, items} = props;
  const location = useLocation();
  const view = getView(location.search);
  const requestedPage = getCurrentPage(location.search);
  const posts = items.filter((item) => !isMeetingMinutes(item));
  const activeItems = view === 'minutes' ? items.filter(isMeetingMinutes) : posts;
  const pageSize = BLOG_POSTS_PER_PAGE;
  const {currentPage, pageItems, totalPages} = paginateItems(
    activeItems,
    requestedPage,
    pageSize,
  );
  const [featured, ...blogPosts] = currentPage === 1 ? pageItems : [];
  const regularBlogPosts = currentPage === 1 ? blogPosts : pageItems;

  return (
    <Layout>
      <main className="container margin-vert--lg">
        <BlogListHeader metadata={metadata} />
        <NewsNav view={view} />
        {view === 'blog' ? (
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <span>Blog</span>
              <h2>Project updates and release notes</h2>
            </div>
            {featured && <FeaturedPost item={featured} />}
            <section className={styles.grid} aria-label="News posts">
              {regularBlogPosts.map((item) => (
                <PostCard
                  key={item.content.metadata.permalink}
                  item={item}
                />
              ))}
            </section>
          </section>
        ) : (
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <span>Meeting Minutes</span>
              <h2>Community meeting notes</h2>
            </div>
            <div className={styles.minutesList}>
              {pageItems.map((item) => (
                <MinutesItem key={item.content.metadata.permalink} item={item} />
              ))}
            </div>
          </section>
        )}
        <ListPagination
          currentPage={currentPage}
          totalPages={totalPages}
          view={view}
        />
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
