<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# ADR-002: Website docs source layout (sync script vs. native loading)

## Status

**Proposed. Draft for discussion; nothing here has been decided.**
Tracking issue: [#1308](https://github.com/apache/mahout/issues/1308).

This file lives in `dev/` rather than `docs/` on purpose: anything under
`/docs` is published on the website, and a proposal should not be.

## Context

`/docs` is the single source of truth for documentation and blog posts.
`website/scripts/sync-docs.js` copies it into the gitignored
`website/docs/` and `website/blog/` before every `npm run start`,
`npm run build` and `npm run version`, and also writes the generated
Python API pages (`qumat/api.md`, `qdp/api.md`) into the copy.

The copy step is deliberate. It came out of the discussion behind #882:
keep `/docs` plain, framework-agnostic markdown, so that the docs are not
tied to Docusaurus.

#1308 asks whether the copy is still worth it. It lists two options:

1. **Patch in place.** Keep the sync and make it robust.
2. **Docusaurus-native.** Point `docs.path` at `/docs` and `blog.path` at
   `/docs/blog`, and delete the sync step.

Option 1 is implemented in the patch series that adds this file:

- Frontmatter is parsed with the same parser Docusaurus uses
  (`@11ty/gray-matter`).
- Generated titles keep acronyms.
- The no-op link transform is removed.
- The script has a `node:test` suite.

The series also fixes the "Edit this page" links. Docusaurus built them
from the copy's path, so all 226 links in a local build pointed at files
that do not exist.

After that series, the sync no longer rewrites the markdown body at all.
It changes only frontmatter: it drops `layout`/`permalink` and fills in a
title when one is missing. Docusaurus tolerates both of those as they are
(unknown frontmatter keys are allowed). This ADR is about whether that
remaining work still justifies keeping a second copy of the docs.

## What a native layout would involve

These findings come from local `docusaurus build` runs (Docusaurus
3.10.2) against commit 2d94f8f. The prototype was not committed.

1. **Docs alone work.** Changing `docs.path` to `'../docs'` with
   `exclude: ['blog/**', 'api/**']` builds. The other 248 HTML pages are
   identical to the synced build. The two generated API pages differ
   only because the prototype used a placeholder for them.

2. **`exclude` applies to every version.** The `api/**` exclude also
   hides `versioned_docs/version-0.6/api/index.md`, so `/docs/api/`
   disappears from the latest release. The generator config in
   `docs/api/` would need to move to a path that is excluded by default
   (for example `docs/_api/`), or out of `/docs` altogether.

3. **The blog cannot stay inside the docs tree.** Setting
   `blog.path: '../docs/blog'` while `docs.path` is `'../docs'` makes
   static rendering fail for all 60 blog routes (posts, index, archive
   and tag pages; `Cannot destructure property 'title' of 'metadata'`). The exclude does not help. As far as I can
   tell from the plugin source, the docs plugin's MDX loader claims every
   file under its content directory, and it treats excluded files as
   partials, so blog posts are compiled without blog metadata. Pointing
   `blog.path` at `'../docs/blog'` *with docs still synced* builds fine.
   So a fully native layout means moving the blog out of `/docs` (for
   example to `/blog`).

4. **Versioning copies the whole docs directory.** `docusaurus
   docs:version X` copies `docs.path` into `versioned_docs/version-X/`
   with `fs.copy` and ignores `exclude`. With `docs.path` at `/docs`,
   anything else under `/docs` (blog, API config) would land in every
   snapshot. This is a second reason to move them out. The generated API
   pages must also exist before the snapshot is taken.
   `scripts/version-docs.js` already runs the sync first. A native
   layout would need a separate "generate API pages" step in its place,
   writing into `/docs` (gitignored paths) or into a second docs plugin
   instance.

5. **Existing snapshots are unaffected.** `website/versioned_docs/` keeps
   working as-is. Versions cut after a switch would contain raw `/docs`
   content, including any `layout:` keys. Docusaurus accepts that
   frontmatter.

6. **Developer loop.** With the sync, a running `npm run start` does
   not see edits made under `/docs` until `npm run sync` is re-run,
   because Docusaurus only watches `website/docs`. Natively, edits
   hot-reload.

## Options

**A. Keep the sync (Option 1, status quo after this series).**
No change to the repo layout, and the #882 decision stands. Costs: a
second copy of the docs, the prebuild hook, the manual re-sync while
running `npm run start`, and the edit-URL mapping in
`docusaurus.config.ts`.

**B. Hybrid: native docs, synced (or moved) blog.**
Set `docs.path: '../docs'`, move the API generator config to
`docs/_api/`, and have the API generator write gitignored files into
`/docs`. The blog either keeps a small copy step or moves to `/blog`.
This keeps most of the benefit with a modest change, but `/docs` gains
gitignored generated files.

**C. Fully native.**
B, plus move the blog to a top-level `/blog`, plus delete
`sync-docs.js` (keeping only the API generator as a script) and the
`website/docs/.gitignore` mirror. This is the cleanest end state. It
changes where contributors put blog posts, and it means `/docs` must be
valid Docusaurus input, which it already is today.

## Decision

None yet. Maintainers to decide in #1308. Points to settle:

- Does the framework-agnostic goal from #882 still need a copy step now
  that the copy no longer changes the markdown body?
- Is moving the blog out of `/docs` acceptable?
- Where should generated API pages live?

## Consequences

To be filled in once an option is chosen.
