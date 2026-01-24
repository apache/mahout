# Apache Mahout Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Quick Start

```bash
cd website-new
npm install
npm run start
```

This starts a local development server at `http://localhost:3000` with hot reload.

## Architecture

### Documentation Source of Truth

**`/docs/`** (at repository root) is the **source of truth** for ALL documentation.

```
/docs/                          ← SOURCE OF TRUTH (edit here!)
├── index.md                    # Main QuMat overview
├── qumat/                      # Qumat documentation
│   ├── core/                   # Qumat Core docs
│   ├── qdp/                    # QDP docs
│   ├── quantum-computing-primer/
│   └── papers/
├── qdp/                        # QDP package internals
├── community/                  # Community pages
├── about/                      # About pages
├── download/                   # Download pages
├── api.md, basic-gates.md...   # API reference docs
└── adr/                        # Architecture decisions

/website-new/docs/              ← BUILD ARTIFACT (auto-synced, don't edit!)
├── .gitignore                  # Only tracked file
└── [everything synced from /docs/]
```

### Sync Workflow

The sync script copies `/docs/` to `website-new/docs/` at build time:

```bash
npm run sync      # Manual sync
npm run start     # Auto-syncs before starting dev server
npm run build     # Auto-syncs before building
```

**Important:**
- Edit ALL documentation in `/docs/`, NOT in `website-new/docs/`
- `website-new/docs/` is gitignored (except `.gitignore` itself)
- Changes in `website-new/docs/` will be overwritten on next sync

## Adding New Documentation

### 1. Add a New Doc Page

Create a markdown file in `/docs/`:

```markdown
---
title: My New Page
---

# My New Page

Content here...
```

### 2. Add to Sidebar

Edit `website-new/sidebars.ts`:

```typescript
const sidebars: SidebarsConfig = {
  docsSidebar: [
    // ... existing items
    {
      type: 'category',
      label: 'My Section',
      items: [
        'my-new-page',           // matches /docs/my-new-page.md
        'folder/another-page',   // matches /docs/folder/another-page.md
      ],
    },
  ],
};
```

### 3. Rebuild

```bash
npm run sync   # Sync changes from /docs/
npm run start  # Start dev server to preview
```

## File Naming Convention

Use **kebab-case** for all documentation files:

```
✅ getting-started.md
✅ quantum-gates.md
✅ basic-gates.md

❌ getting_started.md
❌ GettingStarted.md
```

This creates clean URLs like `/docs/getting-started`.

## Frontmatter

Every doc should have frontmatter:

```markdown
---
title: Page Title              # Required: displayed in browser tab
sidebar_label: Short Label     # Optional: shorter name for sidebar
sidebar_position: 1            # Optional: order in auto-generated sidebars
---
```

## Adding Images

1. Place images in `/docs/assets/` (for code docs) or `website-new/static/img/` (for website)
2. Reference in markdown:

```markdown
![Alt text](./assets/my-image.png)        # Relative path (in /docs/)
![Alt text](/img/my-image.png)            # Absolute path (in static/)
```

## Math/LaTeX

KaTeX is enabled for math rendering:

```markdown
Inline math: $E = mc^2$

Block math:
$$
\ket{\psi} = \alpha\ket{0} + \beta\ket{1}
$$
```

## Code Blocks

Syntax highlighting is available for: Python, Java, Scala, Rust, Bash, and more.

````markdown
```python
from qumat import QumatCircuit

circuit = QumatCircuit(2)
circuit.h(0)
```
````

## Versioning

### Current Versions

- **0.5-dev** - Current development version (synced from `/docs/`)
- **0.4** - Stable release snapshot

### Creating a New Version

When releasing, snapshot the current docs:

```bash
npm run docusaurus docs:version 0.5
```

This creates:
- `versioned_docs/version-0.5/` - Frozen snapshot
- Updates `versions.json`

### Version Configuration

Edit `docusaurus.config.ts`:

```typescript
docs: {
  lastVersion: 'current',
  versions: {
    current: {
      label: '0.6-dev',
      path: '',
    },
    '0.5': {
      label: '0.5',
      path: '0.5',
    },
  },
}
```

## Blog Posts

Blog posts live in `website-new/blog/`:

```markdown
---
title: My Blog Post
date: 2026-01-22
tags: [news, release]
---

Content here...

<!--truncate-->

More content after the fold...
```

## Building for Production

```bash
npm run build    # Creates production build in /build
npm run serve    # Serve production build locally
```

## Deployment

GitHub Actions automatically deploys to `asf-site` branch when changes are pushed to `main`.

Triggers:
- Changes to `website-new/**`
- Changes to `docs/**`
- Changes to `qdp/docs/**`

## Troubleshooting

### Broken Links

The build will warn about broken links. Fix them in the source files (`/docs/`).

### Sidebar Not Updating

1. Run `npm run sync` to re-sync from `/docs/`
2. Check `sidebars.ts` for correct document IDs
3. Document IDs are based on file path without `.md` extension

### Doc ID Format

Document IDs are derived from file paths:
- `/docs/getting-started.md` → `getting-started`
- `/docs/qumat/core/api.md` → `qumat/core/api`
- Numeric prefixes are stripped: `001-migration.md` → `migration`

## Project Structure

```
website-new/
├── docusaurus.config.ts    # Main configuration
├── sidebars.ts             # Sidebar navigation
├── package.json
├── scripts/
│   └── sync-docs.js        # Documentation sync script
├── docs/                   # Synced docs + website-only content
├── versioned_docs/         # Version snapshots
├── blog/                   # Blog posts
├── src/
│   ├── components/         # React components
│   ├── css/custom.css      # Theme customization
│   └── pages/              # Custom pages (homepage)
└── static/                 # Static assets (images, etc.)
```

## Useful Commands

| Command | Description |
|---------|-------------|
| `npm run start` | Start dev server with hot reload |
| `npm run build` | Build production site |
| `npm run serve` | Serve production build locally |
| `npm run sync` | Sync docs from /docs/ |
| `npm run docusaurus docs:version X.Y` | Create version snapshot |
| `npm run clear` | Clear Docusaurus cache |

## Resources

- [Docusaurus Documentation](https://docusaurus.io/docs)
- [Markdown Features](https://docusaurus.io/docs/markdown-features)
- [Sidebar Configuration](https://docusaurus.io/docs/sidebar)
- [Versioning](https://docusaurus.io/docs/versioning)
