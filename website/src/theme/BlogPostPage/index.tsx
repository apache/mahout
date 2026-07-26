import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import {HtmlClassNameProvider, ThemeClassNames} from '@docusaurus/theme-common';
import {
  BlogPostProvider,
  useBlogPost,
} from '@docusaurus/plugin-content-blog/client';
import Layout from '@theme/Layout';
import BlogPostItem from '@theme/BlogPostItem';
import BlogPostPaginator from '@theme/BlogPostPaginator';
import BlogPostPageMetadata from '@theme/BlogPostPage/Metadata';
import BlogPostPageStructuredData from '@theme/BlogPostPage/StructuredData';
import TOC from '@theme/TOC';
import ContentVisibility from '@theme/ContentVisibility';
import type {Props} from '@theme/BlogPostPage';
import blogPostList from '~blog/default/blog-post-list-prop-default.json';

import styles from './styles.module.css';

type BlogListEntry = (typeof blogPostList.items)[number];

function isMeetingMinutes(item: Pick<BlogListEntry, 'permalink' | 'title'>) {
  return (
    item.title.toLowerCase().includes('meeting minutes') ||
    item.permalink.toLowerCase().includes('meeting-minutes')
  );
}

function getSiblingPosts({
  permalink,
  title,
}: Pick<BlogListEntry, 'permalink' | 'title'>) {
  const currentIsMinutes = isMeetingMinutes({permalink, title});
  const peers = blogPostList.items.filter(
    (item) => isMeetingMinutes(item) === currentIsMinutes,
  );
  const currentIndex = peers.findIndex((item) => item.permalink === permalink);

  if (currentIndex === -1) {
    return {nextItem: undefined, prevItem: undefined};
  }

  return {
    prevItem: peers[currentIndex - 1],
    nextItem: peers[currentIndex + 1],
  };
}

function BlogPostPageContent({
  children,
}: {
  children: ReactNode;
}): ReactNode {
  const {metadata, toc} = useBlogPost();
  const {frontMatter, permalink, title} = metadata;
  const {nextItem, prevItem} = getSiblingPosts({permalink, title});
  const {
    hide_table_of_contents: hideTableOfContents,
    toc_min_heading_level: tocMinHeadingLevel,
    toc_max_heading_level: tocMaxHeadingLevel,
  } = frontMatter;

  return (
    <Layout>
      <main className={styles.pageShell}>
        <div className={styles.articleGrid}>
          <div className={styles.articleColumn}>
            <ContentVisibility metadata={metadata} />

            <BlogPostItem>{children}</BlogPostItem>

            {(nextItem || prevItem) && (
              <div className={styles.paginator}>
                <BlogPostPaginator nextItem={nextItem} prevItem={prevItem} />
              </div>
            )}
          </div>
          {!hideTableOfContents && toc.length > 0 && (
            <aside className={styles.tocPanel}>
              <div className={styles.tocTitle}>On this page</div>
              <TOC
                toc={toc}
                minHeadingLevel={tocMinHeadingLevel}
                maxHeadingLevel={tocMaxHeadingLevel}
              />
            </aside>
          )}
        </div>
      </main>
    </Layout>
  );
}

export default function BlogPostPage(props: Props): ReactNode {
  const BlogPostContent = props.content;
  return (
    <BlogPostProvider content={props.content} isBlogPostPage>
      <HtmlClassNameProvider
        className={clsx(
          ThemeClassNames.wrapper.blogPages,
          ThemeClassNames.page.blogPostPage,
          styles.blogPostPage,
        )}>
        <BlogPostPageMetadata />
        <BlogPostPageStructuredData />
        <BlogPostPageContent>
          <BlogPostContent />
        </BlogPostPageContent>
      </HtmlClassNameProvider>
    </BlogPostProvider>
  );
}
