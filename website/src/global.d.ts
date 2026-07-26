/**
 * Type declarations for non-TS modules (e.g. CSS modules).
 * Enables IDE and tsc to recognize imports like: import styles from './styles.module.css'
 */
declare module '*.module.css' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

declare module '~blog/default/blog-post-list-prop-default.json' {
  const value: {
    readonly items: readonly {
      readonly date: string;
      readonly permalink: string;
      readonly title: string;
      readonly unlisted?: boolean;
    }[];
  };
  export default value;
}
