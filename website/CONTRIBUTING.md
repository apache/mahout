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

# Contributing to the Apache Mahout Website

This guide covers **website and documentation** development. For repository-wide workflow (issues, branches, pull requests, pre-commit), see the root [CONTRIBUTING.md](../CONTRIBUTING.md).

## Prerequisites

- Node.js and npm (for Docusaurus)

## Quick Start

From the **`website/`** directory:

```bash
npm install
npm run start
```

This starts the dev server at `http://localhost:3000` with hot reload.

## Documentation Source of Truth

**Edit documentation in `/docs/`** (at repository root), **not** inside `website/docs/`. The website syncs from `/docs/` at build and dev-server start.

- **Source:** `/docs/` — edit here
- **Synced:** `website/docs/` — generated; do not edit manually

Sync manually if needed:

```bash
cd website
npm run sync
```

## Common Tasks

| Task | Command (from `website/`) |
|------|---------------------------|
| Local development | `npm run start` |
| Production build | `npm run build` |
| Sync docs from `/docs/` | `npm run sync` |
| Create version snapshot (release) | `npm run docusaurus docs:version X.Y` |

## Adding or Changing Docs

1. Add or edit Markdown under **`/docs/`** (repo root).
2. Update **`website/sidebars.ts`** if you add new pages or sections.
3. Run `npm run sync` (or restart `npm run start`) to preview.

Use **kebab-case** for file names and frontmatter `title` in each doc. See [website/README.md](README.md) for sidebar config, images, math, and versioning.

## References

- [website/README.md](README.md) — Full website guide: sync, sidebars, versioning, blog, deployment
- [Docusaurus docs](https://docusaurus.io/docs)
