# Jekyll to Docusaurus Migration Plan

## Overview

Migrate the Apache Mahout website from Jekyll 4.3.2 to Docusaurus 3.x, implementing a documentation sync workflow where `/docs` is the source of truth.

## Current State

| Component | Current Technology |
|-----------|-------------------|
| Framework | Jekyll 4.3.2 (Ruby) |
| Styling | Bootstrap 5 + custom SCSS |
| Math | MathJax 3 (self-hosted) |
| Markdown | Kramdown |
| Hosting | ASF infrastructure via `asf-site` branch |
| CI/CD | GitHub Actions |

**Content inventory:**
- 51 blog posts in `/website/_posts/`
- 22+ Qumat documentation pages in `/website/qumat/`
- 8 markdown files in `/docs/` (proposed source of truth)
- 4 papers in `/website/_papers/`
- 3 QDP docs in `/qdp/docs/`

## Decision Summary

| Decision | Choice |
|----------|--------|
| Math rendering | KaTeX (faster, lighter) |
| Migration approach | New directory (`website-new/`) then rename |
| Documentation versioning | Enabled from start |

## Target Architecture

```
website-new/                    # During migration (rename to website/ after)
├── docusaurus.config.js       # Main configuration
├── sidebars.js                # Sidebar configuration
├── package.json
├── scripts/
│   └── sync-docs.js           # Syncs /docs → website-new/docs
├── docs/                      # Current version (synced from /docs)
├── versioned_docs/            # Previous versions
│   └── version-0.4/           # Snapshot of v0.4 docs
├── versioned_sidebars/        # Sidebars for each version
├── versions.json              # Version manifest
├── blog/                      # Migrated from _posts
├── src/
│   ├── components/            # React components (navbar, footer, hero)
│   ├── css/custom.css         # Migrated styles
│   └── pages/index.js         # Custom homepage
└── static/                    # Images, API docs, vendor assets
```

## Implementation Phases

### Phase 1: Foundation Setup

**Tasks:**
1. Initialize Docusaurus 3.x project in `/website-new/`
2. Configure `docusaurus.config.js`:
   - Site metadata (title, URL, baseUrl)
   - Theme colors (primary: `#00bcd4`)
   - Math support via KaTeX (remark-math + rehype-katex)
   - Versioning configuration (current = 0.5-dev, initial snapshot = 0.4)
3. Create `scripts/sync-docs.js` based on gofannon pattern:
   - Copy markdown from `/docs` → `website-new/docs/`
   - Transform frontmatter (Jekyll → Docusaurus format)
   - Handle relative link transformations
4. Set up `package.json` with sync integration:
   ```json
   "scripts": {
     "sync": "node scripts/sync-docs.js",
     "prebuild": "npm run sync",
     "start": "npm run sync && docusaurus start",
     "version": "docusaurus docs:version"
   }
   ```
5. Create initial version snapshot for v0.4 (current stable)

**Files to create:**
- `website-new/docusaurus.config.js`
- `website-new/sidebars.js`
- `website-new/package.json`
- `website-new/scripts/sync-docs.js`
- `website-new/versions.json`

### Phase 2: Content Migration

**Tasks:**
1. Migrate static assets from `/website/assets/` → `website-new/static/`
2. Create blog post migration script (`scripts/migrate-posts.js`):
   - Transform Jekyll frontmatter to Docusaurus format
   - Convert `category` → `tags`
   - Preserve dates and content
3. Run migration for all 51 blog posts
4. Migrate papers collection to custom pages or docs section
5. Configure sidebar structure in `sidebars.js`

**Frontmatter transformation:**
```yaml
# Jekyll (before)                    # Docusaurus (after)
layout: post                         title: Example Post
title: Example Post          →       date: 2025-04-17
date: 2025-04-17                     tags: [news]
category: news                       authors: [mahout-team]
```

### Phase 3: Layout & Styling

**Tasks:**
1. Migrate navbar structure to `docusaurus.config.js`:
   - About dropdown
   - Qumat dropdown (Core, QDP, Primer, Papers)
   - Download, Community, News links
   - GitHub icon
2. Create custom homepage component (`src/pages/index.js`):
   - Hero section with gradient background
   - Feature cards
   - Download CTA
3. Migrate SCSS to CSS custom properties in `src/css/custom.css`:
   ```css
   :root {
     --ifm-color-primary: #00bcd4;
     --ifm-font-family-base: 'Muli', sans-serif;
   }
   ```
4. Configure footer with ASF copyright

**Key files to reference:**
- `/website/_includes/navbar.html` → navbar config
- `/website/_sass/mahout.scss` → CSS variables
- `/website/_layouts/home.html` → homepage structure

### Phase 4: Documentation Integration

**Tasks:**
1. Verify sync workflow copies all `/docs` content correctly
2. Integrate QDP documentation from `/qdp/docs/`
3. Migrate existing `/website/qumat/` documentation
4. Configure Quantum Computing Primer sidebar navigation
5. Test all internal documentation links
6. Verify LaTeX/math rendering in all docs (e.g., `basic_gates.md`)

### Phase 5: Deployment & Cutover

**Tasks:**
1. Update GitHub Actions workflow (`.github/workflows/website.yml`):
   ```yaml
   - uses: actions/setup-node@v4
     with:
       node-version: '18'
   - run: npm ci
     working-directory: website-new
   - run: npm run build
     working-directory: website-new
   - name: Deploy to asf-site
     run: |
       git config user.name "GitHub Actions Bot"
       git config user.email "<>"
       git checkout asf-site
       rm -rf *
       cp -r website-new/build/* .
       git add .
       git commit -m "Automatic Site Publish by Buildbot"
       git push
   ```
2. Configure URL redirects using `@docusaurus/plugin-client-redirects`:
   - `/news.html` → `/blog`
   - Handle `.html` extension removal
3. Run link checker on built site
4. Test deployment to `asf-site` branch
5. Rename `website-new/` → `website/` (archive old Jekyll site)

### Phase 6: Documentation & Cleanup

**Tasks:**
1. Create ADR at `/docs/adr/001-jekyll-to-docusaurus-migration.md`
2. Update `CONTRIBUTING.md` with new website development instructions
3. Remove old Jekyll configuration files
4. Update README with new local development commands

## Versioning Configuration

Docusaurus versioning will be configured to support multiple documentation versions:

**Initial setup:**
```javascript
// docusaurus.config.js
docs: {
  lastVersion: 'current',
  versions: {
    current: {
      label: '0.5-dev',
      path: '',
    },
    '0.4': {
      label: '0.4',
      path: '0.4',
    },
  },
}
```

**Creating a new version:**
```bash
npm run docusaurus docs:version 0.4
```
This snapshots the current `/docs` to `/versioned_docs/version-0.4/`.

**Version workflow:**
1. `/docs` always contains the "next" (development) version
2. When releasing, run `docs:version X.Y` to snapshot
3. Users can switch versions via dropdown in navbar
4. Sync script only updates current `/docs`, versioned docs are static snapshots

## Sync Script Design

Based on the [gofannon sync_docs.py](https://github.com/The-AI-Alliance/gofannon/blob/2eaf7d22a80f4ed8794c02b074ccda137b51ea0b/website/scripts/sync_docs.py) reference, the sync script will:

```javascript
// scripts/sync-docs.js
const SYNC_SOURCES = [
  { source: '../docs', dest: './docs', prefix: '' },
  { source: '../qdp/docs', dest: './docs/qdp', prefix: 'qdp/' },
];

// For each source:
// 1. Clean destination directory
// 2. Copy markdown files recursively
// 3. Transform frontmatter (add sidebar_position, remove layout)
// 4. Convert relative links to Docusaurus format
// 5. Copy associated assets (images)
```

## URL Preservation Strategy

| Content | Old URL | New URL | Redirect |
|---------|---------|---------|----------|
| Homepage | `/` | `/` | None |
| Blog listing | `/news.html` | `/blog` | Yes |
| Blog posts | `/YYYY/MM/DD/title.html` | `/blog/YYYY/MM/DD/title` | Automatic |
| Qumat docs | `/qumat/core/` | `/docs/qumat/core` | Yes |
| Papers | `/qumat/papers/:name/` | `/docs/papers/:name` | Yes |

## Verification Checklist

- [ ] All 51 blog posts migrated and rendering correctly
- [ ] Math/LaTeX displays properly via KaTeX (test with `basic_gates.md`)
- [ ] Navbar matches current structure with version dropdown
- [ ] Homepage hero section renders
- [ ] Sync script copies `/docs` content on build
- [ ] Version switching works (0.4 ↔ 0.5-dev)
- [ ] All internal links resolve in both versions
- [ ] Redirects work for old URLs
- [ ] GitHub Actions deploys successfully to `asf-site`
- [ ] Lighthouse performance score > 90
- [ ] `website-new/` renamed to `website/` and old Jekyll archived

## Key Dependencies

```json
{
  "@docusaurus/core": "^3.0.0",
  "@docusaurus/preset-classic": "^3.0.0",
  "@docusaurus/plugin-client-redirects": "^3.0.0",
  "remark-math": "^3.0.0",
  "rehype-katex": "^5.0.0"
}
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Broken links | Run link checker in CI, comprehensive redirects |
| Math rendering differences | Test all LaTeX content; KaTeX covers 95%+ of MathJax syntax. Audit `basic_gates.md` and quantum primer for edge cases. |
| SEO impact | Maintain URL structure, proper 301 redirects, sitemap |
| Content loss | Keep Jekyll site archived in git history until migration verified |
| Versioning complexity | Start with just 0.4 + current; add versions incrementally as releases happen |
| Version URL changes | Configure `path` options to maintain clean URLs (`/docs/0.4/...`) |

## Architecture Decision Record (ADR)

An ADR should be created at `/docs/adr/001-jekyll-to-docusaurus-migration.md` documenting:

- **Status:** Accepted
- **Context:** Current Jekyll setup challenges (Ruby dependency management, manual doc sync, limited modern features)
- **Decision:** Migrate to Docusaurus 3.x with sync workflow
- **Rationale:** React-based, built-in versioning/search, purpose-built for technical docs, active community
- **Consequences:** Initial migration effort, team learns React/Docusaurus, but simplified long-term maintenance

---

## Work Log - January 22, 2026

### Session 1: Initial Docusaurus Setup & Content Migration

**Completed Tasks:**

1. **Initialized Docusaurus 3.x project** in `/website-new/`
   - Created `docusaurus.config.ts` with TypeScript
   - Configured KaTeX for math rendering (remark-math + rehype-katex)
   - Set up navbar with dropdowns (About, Qumat, Download, Community, News)
   - Configured footer with ASF copyright
   - Added Prism syntax highlighting for Python, Java, Scala, Rust, Bash

2. **Created documentation sync script** (`scripts/sync-docs.js`)
   - Syncs markdown from `/docs` to `website-new/docs/`
   - Transforms Jekyll frontmatter to Docusaurus format
   - Removes `layout:` field, preserves titles
   - Handles link transformations (removes `.md` extensions)

3. **Migrated blog posts** (51 posts)
   - Created `scripts/migrate-posts.js`
   - Transformed Jekyll frontmatter (`category` → `tags`)
   - Preserved dates and content
   - Fixed MDX parsing issues with `markdown: { format: 'detect' }`

4. **Created WaveAnimation component** (`src/components/WaveAnimation/`)
   - Ported jQuery sine wave animation to React
   - Canvas-based animation matching original Jekyll site hero
   - Responsive sizing

5. **Built custom homepage** (`src/pages/index.tsx`)
   - Hero section with wave animation and gradient background
   - Mahout logo (120px) and ASF logo (100px)
   - Download button and version text
   - Sidebar with release info, ASF links, and talks

6. **Migrated static assets**
   - Logos: `mahout-logo-white.svg`, `mahout-logo-blue.svg`, `asf_new_logo.svg`
   - Mascot image: `mascot_with_text.png`
   - Favicon

7. **Updated GitHub Actions workflow** (`.github/workflows/website.yml`)
   - Changed from Ruby/Jekyll to Node.js 20/Docusaurus
   - Added path triggers for `website-new/`, `docs/`, `qdp/docs/`
   - Deploy to `asf-site` branch

8. **Configured URL redirects**
   - `/news.html` → `/blog`
   - `/news` → `/blog`

9. **Created ADR document** (`/docs/adr/001-jekyll-to-docusaurus-migration.md`)
   - Documented migration rationale and decisions
   - Architecture overview
   - Versioning strategy

10. **Updated CONTRIBUTING.md**
    - Added website development instructions
    - Documented sync workflow
    - Added version snapshot instructions

### Session 2: Documentation Restructuring

**Problem Identified:**
The initial migration created a fragmented structure:
- `/docs/` at root had only 8 sparse API files
- `/website/qumat/` had 22+ documentation pages
- `/website-new/docs/` had mixed content from multiple sources
- No clear source of truth

**Solution Implemented:**

1. **Restructured `/docs/` as single source of truth**

   New structure:
   ```
   /docs/
   ├── index.md                     # Main Qumat overview
   ├── api.md                       # API reference
   ├── basic-gates.md               # Gates documentation
   ├── getting-started.md           # Getting started guide
   ├── getting-started-with-qumat.md
   ├── parameterized-circuits.md    # PQC guide
   ├── pqc.md                       # In-depth PQC
   ├── qumat-gap-analysis-for-pqc.md
   ├── adr/
   │   └── 001-jekyll-to-docusaurus-migration.md
   ├── qdp/                         # QDP package docs
   │   ├── observability.md
   │   ├── readers.md
   │   └── testing.md
   └── qumat/
       ├── index.md
       ├── core/
       │   ├── index.md
       │   ├── api.md
       │   ├── concepts.md
       │   ├── examples.md
       │   └── getting-started.md
       ├── qdp/
       │   ├── index.md
       │   ├── api.md
       │   ├── concepts.md
       │   ├── examples.md
       │   └── getting-started.md
       ├── quantum-computing-primer/
       │   ├── index.md
       │   ├── introduction.md
       │   ├── qubits.md
       │   ├── quantum-gates.md
       │   ├── quantum-circuits.md
       │   ├── quantum-entanglement.md
       │   ├── quantum-algorithms.md
       │   ├── quantum-error-correction.md
       │   ├── applications.md
       │   └── advanced-topics.md
       └── papers/
           ├── index.md
           └── [4 paper summaries]
   ```

2. **Moved content from multiple sources:**
   - `/website/qumat/` → `/docs/qumat/`
   - `/website/_papers/` → `/docs/qumat/papers/`
   - `/qdp/docs/` → `/docs/qdp/`

3. **Renamed files to kebab-case:**
   - `basic_gates.md` → `basic-gates.md`
   - `getting_started.md` → `getting-started.md`
   - `parameterized_circuits.md` → `parameterized-circuits.md`
   - `p_q_c.md` → `pqc.md`
   - Primer chapters: `01_introduction/index.md` → `introduction.md`

4. **Cleaned up `website-new/docs/`:**
   - Deleted all synced content
   - Preserved website-only directories:
     - `community/` (code of conduct, mailing lists, who we are)
     - `about/` (how to contribute)
     - `download/` (releases, quickstart)

5. **Updated sync script** (`scripts/sync-docs.js`):
   - Single source: syncs only from `/docs/`
   - Preserves website-only directories during clean
   - No longer syncs from `/qdp/docs/` separately (content moved to `/docs/qdp/`)

6. **Deleted old version snapshot:**
   - Removed `versioned_docs/version-0.4/` (based on old messy structure)
   - Removed `versions.json`
   - Updated `docusaurus.config.ts` to remove 0.4 version config

7. **Verified build succeeds:**
   - `npm run sync` - Synced 40 files, 8 directories
   - `npm run build` - SUCCESS with broken link warnings
   - Broken links are due to incorrect paths in source docs (e.g., `/qumat/core` should be `./core`)

### Current State

| Component | Status |
|-----------|--------|
| Docusaurus setup | ✅ Complete |
| Blog migration (51 posts) | ✅ Complete |
| Homepage with wave animation | ✅ Complete |
| `/docs/` restructured | ✅ Complete |
| Sync script | ✅ Complete |
| GitHub Actions | ✅ Complete |
| ADR document | ✅ Complete |
| Build | ✅ Succeeds |

### Remaining Work

1. **Fix broken internal links in `/docs/`:**
   - Links like `/qumat/core` should be `./core` (relative)
   - Primer links like `01_introduction/` should be `./introduction`

2. **Create version 0.4 snapshot** (after links are fixed):
   ```bash
   npm run docusaurus docs:version 0.4
   ```

3. **Final cutover:**
   - Test deployment to `asf-site` branch
   - Rename `website-new/` → `website/`
   - Archive old Jekyll site

### Files Created/Modified Today

| File | Action |
|------|--------|
| `/docs/qumat/**` | Created - moved from /website/qumat/ |
| `/docs/qdp/**` | Created - moved from /qdp/docs/ |
| `/docs/*.md` | Renamed to kebab-case |
| `website-new/scripts/sync-docs.js` | Updated - new sync logic |
| `website-new/docusaurus.config.ts` | Updated - removed 0.4 version |
| `website-new/docs/` | Cleaned - only website-only content remains |
| `.github/workflows/website.yml` | Updated - Node.js/Docusaurus |
| `CONTRIBUTING.md` | Updated - website dev instructions |
| `/docs/adr/001-jekyll-to-docusaurus-migration.md` | Created |
