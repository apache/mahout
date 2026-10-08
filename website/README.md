# Apache Mahout Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Prerequisites

- Node.js
- npm (for Docusaurus)

## Quick Start

```bash
cd website
npm install
npm run start
```

This starts a local development server at `http://localhost:3000` with hot reload. The changes to the documentation will be reflected at `http://localhost:3000/docs/master/`.

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
└── api.md, basic-gates.md...   # API reference docs

/website/docs/              ← BUILD ARTIFACT (auto-synced, don't edit!)
├── .gitignore                  # Only tracked file
└── [everything synced from /docs/]
```

### Sync Workflow

The sync script copies `/docs/` to `website/docs/` at build time:

```bash
npm run sync      # Manual sync
npm run start     # Auto-syncs before starting dev server
npm run build     # Auto-syncs before building
```

**Important:**
- Edit ALL documentation in `/docs/`, NOT in `website/docs/`
- `website/docs/` is gitignored (except `.gitignore` itself)
- Changes in `website/docs/` will be overwritten on next sync

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

Edit `website/sidebars.ts`:

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

1. Place images in `/docs/assets/` (for code docs) or `website/static/img/` (for website)
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
from qumat import QuMat

qumat = QuMat({
    "backend_name": "qiskit",
    "backend_options": {"simulator_type": "aer_simulator"},
})
qumat.create_empty_circuit(num_qubits=2)
qumat.apply_hadamard_gate(0)
```
````

## Versioning

### Current Versions

Released versions are listed in `versions.json`. The first version in that file is
served as `latest` at `/docs`; the editable current docs are served at `/docs/master`.

### Creating a New Version

When releasing, snapshot the current docs:

```bash
npm run version -- <release-version>
```

This creates:
- `versioned_docs/version-<release-version>/` - Frozen snapshot
- Updates `versions.json`
- Validates the generated versioned docs with a production Docusaurus build

`docusaurus.config.ts` reads `versions.json` automatically, so release updates
should not require editing the Docusaurus version configuration.

The command runs the documentation sync first, so generated Python API reference
pages are included in the version snapshot before the local build validation.

## API Reference Docs

The Python API reference pages are generated from docstrings; they are
not written by hand.

| Page | Generator config | Header (frontmatter + intro) | Output (gitignored) |
|------|------------------|------------------------------|---------------------|
| Qumat | `docs/api/qumat.yml` | `docs/api/qumat_header.md` | `website/docs/qumat/api.md` |
| QDP | `docs/api/qumat_qdp.yml` | `docs/api/qumat_qdp_header.md` | `website/docs/qdp/api.md` |

`npm run sync` (and therefore `npm run start` and `npm run build`) runs
`uv sync --frozen --group dev`, then
`uv run --frozen --group dev pydoc-markdown docs/api/<config>.yml` for
each page, and writes the header followed by the generated markdown.

- **To change what a page says**, edit the docstrings in `qumat/` or
  `qdp/qdp-python/qumat_qdp/`.
- **To change which modules or members appear**, edit the `modules` list
  or the `filter` processor in the `.yml` file. `qumat.yml` hides
  private names; `qumat_qdp.yml` uses `documented_only: true`, so a QDP
  class or method without a docstring is left out.
- **To change the page title, sidebar label or intro**, edit the
  `*_header.md` file.

pydoc-markdown parses the source statically. It does not import `qumat`,
`qumat_qdp`, `torch` or the Rust extension, so API pages can be
generated on a laptop with no GPU and no `maturin develop`.

### Preview locally

```bash
cd website
npm run start          # regenerate API pages, then serve /docs/master/
npm run start:no-api   # skip generation; API pages show a placeholder
```

Open `http://localhost:3000/docs/master/qumat/api` and
`http://localhost:3000/docs/master/qdp/api`. Docusaurus does not watch
`/docs` or the Python sources, so after editing a docstring run
`npm run sync` again; the running dev server picks up the new files.

To see only the generated markdown for one page (handy while editing a
docstring), run from the repository root:

```bash
uv run --group dev pydoc-markdown docs/api/qumat_qdp.yml | less
```

### Rust API docs

The Rust crates in `qdp/` are not published on the website. To browse
their API locally:

```bash
cd qdp
cargo doc -p qdp-core --no-deps --open
```

This builds without a GPU or CUDA toolkit: the build scripts fall back
to stub CUDA symbols and print `CUDA toolkit not found` warnings, which
are expected. Limitations to keep in mind:

- Much of `qdp-core` is behind `#[cfg(target_os = "linux")]`
  (`QdpEngine` methods, the encoders, the pipeline runner). On macOS or
  Windows those items are missing from the generated docs; build on
  Linux, in the dev container, or under WSL to see the full API.
- Items behind Cargo features (for example `remote-io`, `pytorch`) only
  appear with `--features <name>`. `pytorch` builds the `tch` crate,
  which needs libtorch; the repo's pre-commit hook gets it from the
  venv's PyTorch via `LIBTORCH_USE_PYTORCH=1`.
- rustdoc currently reports warnings for `qdp-core` (unescaped `<f64>`
  style generics and math such as `[i]` read as links). They do not
  fail the build; fixes are welcome.

### In CI and on the live site

- The **Website Build Check** workflow runs `npm run build` on pull
  requests that touch `website/`, `docs/`, `qumat/` or
  `qdp/qdp-python/qumat_qdp/`, so docstring-only changes are built too.
  It does not upload a preview; check the build log, or build locally
  and run `npm run serve`.
- The **Deploy Website** workflow runs on pushes to `main` that touch
  `website/` or `docs/`. A change to docstrings alone does not trigger a
  deploy; the new API pages go live with the next website deploy (or a
  manual `workflow_dispatch` run).
- Broken links are reported as warnings, not errors
  (`onBrokenLinks: 'warn'`). Read the "broken links" list in the build
  output for pages you touched.

### Versioning

`npm run version -- <release-version>` runs the sync first, so the
generated `qumat/api.md` and `qdp/api.md` are copied into
`versioned_docs/version-<release-version>/` and frozen with the
release. Do not use `npm run sync:no-api` before cutting a version, or
the snapshot will contain the placeholder pages. After versioning,
check that the snapshot contains both API pages and that the build
reports no new broken links under `/docs/<release-version>/`.

### Troubleshooting

- **`Error: spawnSync uv ENOENT`** during `npm run sync`, `start` or
  `build`: `uv` is not installed or not on `PATH`. Install it (see the
  root `CONTRIBUTING.md`), or use `npm run start:no-api` if you do not
  need the API pages.
- **API page says "API reference generation was skipped"**: it was
  synced with `sync:no-api` / `start:no-api`. Run `npm run sync`.
- **A class or method is missing from the QDP page**: it has no
  docstring (`documented_only: true`), its module is not in the
  `modules` list in `docs/api/qumat_qdp.yml`, or its name starts with
  `_`.
- **Docstring edit not showing in `npm run start`**: re-run
  `npm run sync`; the dev server does not regenerate API pages.

## Blog Posts

Blog posts live in `website/blog/`:

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
- Changes to `website/**`
- Changes to `docs/**`
- Changes to `qdp/docs/**`

## Troubleshooting

### Changes are not reflected in the development server

Please ensure that you are looking at `http://localhost:3000/docs/master/`.

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
website/
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
