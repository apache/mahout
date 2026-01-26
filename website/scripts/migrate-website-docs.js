#!/usr/bin/env node

/**
 * Website Documentation Migration Script for Apache Mahout
 *
 * Migrates documentation from the Jekyll website structure to Docusaurus.
 * This handles:
 * - /website/qumat/ → /website-new/docs/qumat/
 * - /website/community/ → /website-new/docs/community/
 * - /website/about/ → /website-new/docs/about/
 * - /website/download/ → /website-new/docs/download/
 */

const fs = require('fs');
const path = require('path');

const WEBSITE_DIR = path.resolve(__dirname, '../../website');
const DOCS_DIR = path.resolve(__dirname, '../docs');

// Directories to migrate
const MIGRATE_DIRS = [
  { source: 'qumat', dest: 'qumat' },
  { source: 'community', dest: 'community' },
  { source: 'about', dest: 'about' },
  { source: 'download', dest: 'download' },
];

// Files/directories to exclude
const EXCLUDE_PATTERNS = [
  /^\./,
  /^node_modules$/,
  /\.pyc$/,
  /^__pycache__$/,
];

function shouldExclude(name) {
  return EXCLUDE_PATTERNS.some(pattern => pattern.test(name));
}

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

/**
 * Parse YAML frontmatter
 */
function parseFrontmatter(content) {
  const frontmatterRegex = /^---\n([\s\S]*?)\n---\n/;
  const match = content.match(frontmatterRegex);

  if (!match) {
    return { frontmatter: {}, body: content };
  }

  const frontmatterStr = match[1];
  const body = content.slice(match[0].length);

  const frontmatter = {};
  frontmatterStr.split('\n').forEach(line => {
    const colonIndex = line.indexOf(':');
    if (colonIndex > 0) {
      const key = line.slice(0, colonIndex).trim();
      let value = line.slice(colonIndex + 1).trim();
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      frontmatter[key] = value;
    }
  });

  return { frontmatter, body };
}

/**
 * Generate frontmatter string
 */
function generateFrontmatter(frontmatter) {
  const lines = ['---'];
  for (const [key, value] of Object.entries(frontmatter)) {
    if (typeof value === 'string' && (value.includes(':') || value.includes('#'))) {
      lines.push(`${key}: "${value}"`);
    } else {
      lines.push(`${key}: ${value}`);
    }
  }
  lines.push('---\n');
  return lines.join('\n');
}

/**
 * Transform frontmatter for Docusaurus
 */
function transformFrontmatter(frontmatter, filePath, relativePath) {
  const transformed = { ...frontmatter };

  // Remove Jekyll-specific fields
  delete transformed.layout;
  delete transformed.permalink;

  // Keep title if present
  if (!transformed.title) {
    const basename = path.basename(filePath, '.md');
    if (basename !== 'index') {
      transformed.title = basename
        .replace(/_/g, ' ')
        .replace(/-/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
    }
  }

  // Add sidebar_label if title exists
  if (transformed.title && !transformed.sidebar_label) {
    transformed.sidebar_label = transformed.title;
  }

  return transformed;
}

/**
 * Transform links in content
 */
function transformLinks(content) {
  // Convert .html links to remove extension
  content = content.replace(/\]\(([^)]+)\.html\)/g, ']($1)');

  // Convert absolute /qumat/ links to /docs/qumat/
  content = content.replace(/\]\(\/qumat\//g, '](/docs/qumat/');
  content = content.replace(/\]\(\/community\//g, '](/docs/community/');
  content = content.replace(/\]\(\/download\//g, '](/docs/download/');
  content = content.replace(/\]\(\/about\//g, '](/docs/about/');

  return content;
}

/**
 * Process a markdown file
 */
function processMarkdownFile(srcPath, destPath, relativePath) {
  let content = fs.readFileSync(srcPath, 'utf-8');

  const { frontmatter, body } = parseFrontmatter(content);
  const transformedFrontmatter = transformFrontmatter(frontmatter, srcPath, relativePath);
  const transformedBody = transformLinks(body);

  let finalContent;
  if (Object.keys(transformedFrontmatter).length > 0) {
    finalContent = generateFrontmatter(transformedFrontmatter) + transformedBody;
  } else {
    finalContent = transformedBody;
  }

  fs.writeFileSync(destPath, finalContent);
}

/**
 * Copy non-markdown files
 */
function copyFile(srcPath, destPath) {
  fs.copyFileSync(srcPath, destPath);
}

/**
 * Recursively migrate a directory
 */
function migrateDirectory(srcDir, destDir, relativePath = '', stats = { files: 0, dirs: 0 }) {
  if (!fs.existsSync(srcDir)) {
    console.log(`  Source does not exist: ${srcDir}`);
    return stats;
  }

  ensureDir(destDir);

  const entries = fs.readdirSync(srcDir, { withFileTypes: true });

  for (const entry of entries) {
    if (shouldExclude(entry.name)) {
      continue;
    }

    const srcPath = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);
    const newRelativePath = path.join(relativePath, entry.name);

    if (entry.isDirectory()) {
      migrateDirectory(srcPath, destPath, newRelativePath, stats);
      stats.dirs++;
    } else if (entry.isFile()) {
      if (entry.name.endsWith('.md')) {
        processMarkdownFile(srcPath, destPath, newRelativePath);
      } else {
        copyFile(srcPath, destPath);
      }
      stats.files++;
    }
  }

  return stats;
}

/**
 * Main migration function
 */
function main() {
  console.log('Starting website documentation migration...\n');

  let totalFiles = 0;
  let totalDirs = 0;

  for (const dir of MIGRATE_DIRS) {
    const srcDir = path.join(WEBSITE_DIR, dir.source);
    const destDir = path.join(DOCS_DIR, dir.dest);

    console.log(`Migrating: ${dir.source}`);
    console.log(`  From: ${srcDir}`);
    console.log(`  To:   ${destDir}`);

    const stats = migrateDirectory(srcDir, destDir);

    console.log(`  Migrated ${stats.files} files, ${stats.dirs} directories\n`);
    totalFiles += stats.files;
    totalDirs += stats.dirs;
  }

  console.log(`Migration complete! Total: ${totalFiles} files, ${totalDirs} directories`);
}

// Run migration
main();
