/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import Link, {type Props as LinkProps} from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import ThemedImage from '@theme/ThemedImage';
import AuthorSocials from '@theme/Blog/Components/Author/Socials';
import type {Props} from '@theme/Blog/Components/Author';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type AuthorImageVariants = {
  imageURLDark?: string;
  image_url_dark?: string;
};

function getAuthorImageURLDark(author: Props['author']): string | undefined {
  const authorWithDark = author as Props['author'] & AuthorImageVariants;
  return authorWithDark.imageURLDark ?? authorWithDark.image_url_dark;
}

function MaybeLink(props: LinkProps): ReactNode {
  if (props.href) {
    return <Link {...props} />;
  }
  return <>{props.children}</>;
}

function AuthorTitle({title}: {title: string}) {
  return (
    <small className={styles.authorTitle} title={title}>
      {title}
    </small>
  );
}

function AuthorName({name, as}: {name: string; as: Props['as']}) {
  if (!as) {
    return (
      <span className={styles.authorName} translate="no">
        {name}
      </span>
    );
  } else {
    return (
      <Heading as={as} className={styles.authorName} translate="no">
        {name}
      </Heading>
    );
  }
}

function AuthorBlogPostCount({count}: {count: number}) {
  return <span className={clsx(styles.authorBlogPostCount)}>{count}</span>;
}

// Note: in the future we might want to have multiple "BlogAuthor" components
// Creating different display modes with the "as" prop may not be the best idea
// Explainer: https://kyleshevlin.com/prefer-multiple-compositions/
// For now, we almost use the same design for all cases, so it's good enough
export default function BlogAuthor({
  as,
  author,
  className,
  count,
}: Props): ReactNode {
  const {name, title, url, imageURL: imageURLRaw, email, page} = author;
  const link =
    page?.permalink || url || (email && `mailto:${email}`) || undefined;
  const imageURL = imageURLRaw ? useBaseUrl(imageURLRaw) : undefined;
  const imageURLDarkRaw = getAuthorImageURLDark(author);
  const imageURLDark = imageURLDarkRaw ? useBaseUrl(imageURLDarkRaw) : undefined;

  const image = imageURL ? (
    imageURLDark ? (
      <ThemedImage
        className={clsx('avatar__photo', styles.authorImage)}
        sources={{
          light: imageURL,
          dark: imageURLDark,
        }}
        alt={name}
      />
    ) : (
      <img
        className={clsx('avatar__photo', styles.authorImage)}
        src={imageURL}
        alt={name}
      />
    )
  ) : null;

  return (
    <div
      className={clsx(
        'avatar margin-bottom--sm',
        className,
        styles[`author-as-${as}`],
      )}>
      {image && (
        <MaybeLink href={link} className="avatar__photo-link">
          {image}
        </MaybeLink>
      )}

      {(name || title) && (
        <div className={clsx('avatar__intro', styles.authorDetails)}>
          <div className="avatar__name">
            {name && (
              <MaybeLink href={link}>
                <AuthorName name={name} as={as} />
              </MaybeLink>
            )}
            {count !== undefined && <AuthorBlogPostCount count={count} />}
          </div>
          {!!title && <AuthorTitle title={title} />}

          {/*
            We always render AuthorSocials even if there's none
            This keeps other things aligned with flexbox layout
          */}
          <AuthorSocials author={author} />
        </div>
      )}
    </div>
  );
}
