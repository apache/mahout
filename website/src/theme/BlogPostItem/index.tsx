import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {translate} from '@docusaurus/Translate';
import {useBlogPost} from '@docusaurus/plugin-content-blog/client';
import {usePluralForm} from '@docusaurus/theme-common';
import {useDateTimeFormat} from '@docusaurus/theme-common/internal';
import BlogPostItemContent from '@theme/BlogPostItem/Content';
import BlogPostItemFooter from '@theme/BlogPostItem/Footer';
import BlogPostItemHeaderAuthors from '@theme/BlogPostItem/Header/Authors';
import type {Props} from '@theme/BlogPostItem';

import styles from './styles.module.css';

function useFormattedDate() {
  const dateTimeFormat = useDateTimeFormat({
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });

  return (blogDate: string) => dateTimeFormat.format(new Date(blogDate));
}

function useReadingTimeLabel() {
  const {selectMessage} = usePluralForm();

  return (readingTimeFloat: number) => {
    const readingTime = Math.ceil(readingTimeFloat);
    return selectMessage(
      readingTime,
      translate(
        {
          id: 'theme.blog.post.readingTime.plurals',
          description:
            'Pluralized label for "{readingTime} min read". Use as much plural forms (separated by "|") as your language support (see https://www.unicode.org/cldr/cldr-aux/charts/34/supplemental/language_plural_rules.html)',
          message: 'One min read|{readingTime} min read',
        },
        {readingTime},
      ),
    );
  };
}

function PostMeta() {
  const {
    metadata: {date, readingTime},
  } = useBlogPost();
  const formatDate = useFormattedDate();
  const readingTimeLabel = useReadingTimeLabel();

  return (
    <div className={styles.meta}>
      <time dateTime={date}>{formatDate(date)}</time>
      {typeof readingTime !== 'undefined' && (
        <>
          <span aria-hidden="true">/</span>
          <span>{readingTimeLabel(readingTime)}</span>
        </>
      )}
    </div>
  );
}

function ListPostItem({children, className}: Props): ReactNode {
  const {
    metadata: {permalink, title},
  } = useBlogPost();

  return (
    <article className={clsx(styles.listItem, className)}>
      <PostMeta />
      <h2 className={styles.listTitle}>
        <Link to={permalink}>{title}</Link>
      </h2>
      <div className={styles.listAuthors}>
        <BlogPostItemHeaderAuthors />
      </div>
      <BlogPostItemContent className={styles.listContent}>
        {children}
      </BlogPostItemContent>
      <BlogPostItemFooter />
    </article>
  );
}

function ArticlePostItem({children, className}: Props): ReactNode {
  const {
    metadata: {tags, title},
  } = useBlogPost();
  const isMeetingMinutes =
    title.toLowerCase().includes('meeting minutes') ||
    tags.some((tag) => tag.label.toLowerCase() === 'minutes');
  const breadcrumb = isMeetingMinutes
    ? {label: 'Meeting Minutes', to: '/blog/minutes'}
    : {label: 'Blog', to: '/blog'};

  return (
    <article className={clsx(styles.article, className)}>
      <header className={styles.articleHeader}>
        <Link className={styles.breadcrumb} to={breadcrumb.to}>
          {breadcrumb.label}
        </Link>
        <PostMeta />
        <h1 className={styles.articleTitle}>{title}</h1>
        <BlogPostItemHeaderAuthors />
      </header>
      <BlogPostItemContent className={styles.articleContent}>
        {children}
      </BlogPostItemContent>
    </article>
  );
}

export default function BlogPostItem(props: Props): ReactNode {
  const {isBlogPostPage} = useBlogPost();

  return isBlogPostPage ? (
    <ArticlePostItem {...props} />
  ) : (
    <ListPostItem {...props} />
  );
}
