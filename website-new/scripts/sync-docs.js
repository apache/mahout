#!/usr/bin/env node

/**
 * Documentation Sync Script for Apache Mahout
 *
 * This script syncs documentation from the source /docs directory
 * into the Docusaurus website. It:
 * 1. Copies markdown files from /docs to website-new/docs
 * 2. Transforms frontmatter for Docusaurus compatibility
 * 3. Preserves website-only directories (community, about, download)
 */

const fs = require('fs');
const path = require('path');

// Configuration
const SOURCE_DIR = path.resolve(__dirname, '../../docs');
const DEST_DIR = path.resolve(__dirname, '../docs');

// Directories that are website-only and should NOT be deleted during sync
const WEBSITE_ONLY_DIRS = ['community', 'about', 'download'];

// Files that should be preserved during sync (not deleted)
const PRESERVE_FILES = ['.gitignore'];

// Files/directories to exclude from sync
const EXCLUDE_PATTERNS = [
  /^\./, // Hidden files
  /^node_modules$/,
  /\.pyc$/,
  /^__pycache__$/,
];

/**
 * Check if a file/directory should be excluded
 */
function shouldExclude(name) {
  return EXCLUDE_PATTERNS.some(pattern => pattern.test(name));
}

/**
 * Ensure directory exists, creating it if necessary
 */
function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

/**
 * Clean a directory but preserve website-only content
 */
function cleanDestination(destDir) {
  if (!fs.existsSync(destDir)) {
    ensureDir(destDir);
    return;
  }

  const entries = fs.readdirSync(destDir, { withFileTypes: true });

  for (const entry of entries) {
    // Skip website-only directories
    if (entry.isDirectory() && WEBSITE_ONLY_DIRS.includes(entry.name)) {
      console.log(`  Preserving website-only: ${entry.name}/`);
      continue;
    }

    // Skip preserved files (like .gitignore)
    if (entry.isFile() && PRESERVE_FILES.includes(entry.name)) {
      console.log(`  Preserving file: ${entry.name}`);
      continue;
    }

    const fullPath = path.join(destDir, entry.name);
    fs.rmSync(fullPath, { recursive: true, force: true });
  }
}

/**
 * Parse YAML frontmatter from markdown content
 */
function parseFrontmatter(content) {
  const frontmatterRegex = /^---\n([\s\S]*?)\n---\n/;
  const match = content.match(frontmatterRegex);

  if (!match) {
    return { frontmatter: {}, body: content };
  }

  const frontmatterStr = match[1];
  const body = content.slice(match[0].length);

  // Simple YAML parsing (key: value pairs)
  const frontmatter = {};
  frontmatterStr.split('\n').forEach(line => {
    const colonIndex = line.indexOf(':');
    if (colonIndex > 0) {
      const key = line.slice(0, colonIndex).trim();
      let value = line.slice(colonIndex + 1).trim();
      // Remove quotes if present
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
 * Generate YAML frontmatter string
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
 * Transform frontmatter for Docusaurus compatibility
 */
function transformFrontmatter(frontmatter, filePath) {
  const transformed = { ...frontmatter };

  // Remove Jekyll-specific fields
  delete transformed.layout;
  delete transformed.permalink;

  // Generate title from filename if not present
  if (!transformed.title) {
    const basename = path.basename(filePath, '.md');
    if (basename !== 'index') {
      transformed.title = basename
        .replace(/_/g, ' ')
        .replace(/-/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
    }
  }

  return transformed;
}

/**
 * Transform markdown links for Docusaurus
 */
function transformLinks(content) {
  // Match markdown links: [text](url)
  const linkRegex = /\[([^\]]*)\]\(([^)]+)\)/g;

  return content.replace(linkRegex, (match, text, url) => {
    // Skip external links
    if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('//')) {
      return match;
    }

    // Skip anchor links
    if (url.startsWith('#')) {
      return match;
    }

    // Handle .md file references
    if (url.endsWith('.md')) {
      // Remove .md extension for Docusaurus
      url = url.slice(0, -3);
    }

    return `[${text}](${url})`;
  });
}

/**
 * Process a single markdown file
 */
function processMarkdownFile(srcPath, destPath) {
  let content = fs.readFileSync(srcPath, 'utf-8');

  const { frontmatter, body } = parseFrontmatter(content);
  const transformedFrontmatter = transformFrontmatter(frontmatter, srcPath);
  const transformedBody = transformLinks(body);

  // Only add frontmatter if there's something to add
  let finalContent;
  if (Object.keys(transformedFrontmatter).length > 0) {
    finalContent = generateFrontmatter(transformedFrontmatter) + transformedBody;
  } else {
    finalContent = transformedBody;
  }

  fs.writeFileSync(destPath, finalContent);
}

/**
 * Copy a file (binary or non-markdown)
 */
function copyFile(srcPath, destPath) {
  fs.copyFileSync(srcPath, destPath);
}

/**
 * Recursively sync a directory
 */
function syncDirectory(srcDir, destDir, stats = { files: 0, dirs: 0 }) {
  if (!fs.existsSync(srcDir)) {
    console.log(`  Source directory does not exist: ${srcDir}`);
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

    if (entry.isDirectory()) {
      syncDirectory(srcPath, destPath, stats);
      stats.dirs++;
    } else if (entry.isFile()) {
      if (entry.name.endsWith('.md')) {
        processMarkdownFile(srcPath, destPath);
      } else {
        copyFile(srcPath, destPath);
      }
      stats.files++;
    }
  }

  return stats;
}

/**
 * Main sync function
 */
function main() {
  console.log('Starting documentation sync...\n');
  console.log(`Source: ${SOURCE_DIR}`);
  console.log(`Destination: ${DEST_DIR}\n`);

  // Clean destination but preserve website-only content
  console.log('Cleaning destination (preserving website-only content)...');
  cleanDestination(DEST_DIR);

  // Sync documentation
  console.log('\nSyncing documentation from /docs...');
  const stats = syncDirectory(SOURCE_DIR, DEST_DIR);

  console.log(`\nSync complete!`);
  console.log(`  Files: ${stats.files}`);
  console.log(`  Directories: ${stats.dirs}`);
}

// Run the sync
main();
