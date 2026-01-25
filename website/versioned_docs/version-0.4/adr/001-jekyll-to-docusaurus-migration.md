---
title: "ADR-001: Jekyll to Docusaurus Migration"
sidebar_label: "ADR-001: Jekyll to Docusaurus"
---

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

# ADR-001: Jekyll to Docusaurus Migration

## Status

Accepted

## Date

2026-01-22

## Context

The Apache Mahout website was built using Jekyll 4.3.2 with Bootstrap 5, MathJax 3 for math rendering, and Kramdown for Markdown processing. As the project evolved with the introduction of Qumat (quantum computing capabilities), we needed to evaluate our documentation infrastructure.

### Problems with Jekyll

1. **Ruby ecosystem maintenance** - Managing Ruby/Bundler dependencies and versions adds complexity
2. **Limited interactivity** - Jekyll is purely static; no React components for dynamic content
3. **Documentation versioning** - No built-in support for multi-version documentation
4. **Developer experience** - Hot reload and local development workflows are slower
5. **Documentation sync** - No standard pattern for syncing `/docs` with website content

### Requirements

- Documentation versioning for Qumat releases (0.4, 0.5, etc.)
- Math/LaTeX support for quantum computing documentation
- `/docs` directory as source of truth for documentation
- Fast local development with hot reload
- React components for enhanced interactivity
- Modern JavaScript tooling (Node.js ecosystem)

## Decision

Migrate from Jekyll to **Docusaurus 3.x** with the following configuration:

### Technology Choices

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Framework | Docusaurus 3.x | Active development, React-based, built-in versioning |
| Math rendering | KaTeX | Faster than MathJax, lighter bundle size, covers 95%+ use cases |
| Migration approach | New directory (`website-new/`) | Safe migration, can keep Jekyll during transition |
| Versioning | Enabled from start | Supports Qumat release cycle |

### Architecture

```
website-new/
├── docusaurus.config.ts      # Main configuration
├── sidebars.ts               # Sidebar navigation
├── package.json
├── scripts/
│   └── sync-docs.js          # Syncs /docs → website-new/docs
├── docs/                     # Current version (synced from /docs)
├── versioned_docs/           # Previous versions (snapshots)
│   └── version-0.4/
├── versions.json             # Version manifest
├── blog/                     # Migrated from Jekyll _posts
├── src/
│   ├── components/           # React components
│   ├── css/custom.css        # Theme customizations
│   └── pages/index.tsx       # Custom homepage
└── static/                   # Images and assets
```

### Documentation Sync Workflow

The `/docs` directory at the repository root is the source of truth. A sync script copies documentation to the website at build time:

1. `npm run sync` - Copies `/docs` and `/qdp/docs` to `website-new/docs/`
2. Transforms frontmatter as needed
3. Runs automatically before `npm run build` and `npm run start`

This pattern allows:
- Documentation to live alongside code
- Website to always reflect latest `/docs`
- Version snapshots to preserve historical documentation

### Versioning Strategy

- `current` (0.5-dev): Active development, synced from `/docs`
- `0.4`: First stable snapshot, frozen documentation

To create a new version:
```bash
npm run docusaurus docs:version X.Y
```

## Consequences

### Positive

1. **Versioned documentation** - Users can view docs for their specific Qumat version
2. **Faster builds** - Node.js build pipeline is faster than Jekyll
3. **React components** - Can add interactive examples, code playgrounds
4. **Better DX** - Fast hot reload, TypeScript support, modern tooling
5. **KaTeX performance** - Math rendering is 10x faster than MathJax
6. **Active community** - Docusaurus has strong Meta backing and active maintenance

### Negative

1. **Learning curve** - Team needs to learn React/Docusaurus patterns
2. **Migration effort** - One-time cost to migrate all content
3. **KaTeX limitations** - Some advanced MathJax features may not work (rare edge cases)

### Neutral

1. **Bundle size** - Slightly larger JavaScript bundle, but better caching
2. **CI/CD changes** - GitHub Actions updated from Ruby to Node.js

## Implementation Notes

### Key Files Changed

- `.github/workflows/website.yml` - Now uses Node.js instead of Ruby
- `website-new/docusaurus.config.ts` - Main Docusaurus configuration
- `website-new/scripts/sync-docs.js` - Documentation sync script
- `website-new/src/components/WaveAnimation/` - Ported hero animation

### Redirect Strategy

Client-side redirects handle old Jekyll URLs:
- `/news.html` → `/blog`

Additional redirects can be added to `docusaurus.config.ts` as needed.

## References

- [Docusaurus documentation](https://docusaurus.io/)
- [KaTeX supported functions](https://katex.org/docs/supported.html)
- [GitHub discussion](https://github.com/apache/mahout/discussions/) (original proposal)
