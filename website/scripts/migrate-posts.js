#!/usr/bin/env node

/**
 * Blog Post Migration Script for Apache Mahout
 *
 * Migrates Jekyll blog posts from website/_posts to Docusaurus blog format.
 * Transforms frontmatter and preserves content.
 */

const fs = require('fs');
const path = require('path');

const SOURCE_DIR = path.resolve(__dirname, '../../website/_posts');
const DEST_DIR = path.resolve(__dirname, '../blog');

// Author definitions for the blog
const AUTHORS = {
  'mahout-team': {
    name: 'Apache Mahout Team',
    url: 'https://github.com/apache/mahout',
  },
};

/**
 * Parse YAML frontmatter from Jekyll post
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
 * Transform Jekyll frontmatter to Docusaurus format
 */
function transformFrontmatter(jekyllFm, filename) {
  const docusaurusFm = {};

  // Title (required)
  if (jekyllFm.title) {
    docusaurusFm.title = jekyllFm.title;
  } else {
    // Generate from filename: 2020-10-30-weekly-meeting-minutes.md -> Weekly Meeting Minutes
    const titlePart = filename.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
    docusaurusFm.title = titlePart
      .replace(/-/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  }

  // Date - extract from filename or frontmatter
  const dateMatch = filename.match(/^(\d{4}-\d{2}-\d{2})/);
  if (dateMatch) {
    docusaurusFm.date = dateMatch[1];
  } else if (jekyllFm.date) {
    // Parse Jekyll date format: 2025-04-17 00:00:00 -0800
    const datePart = jekyllFm.date.split(' ')[0];
    docusaurusFm.date = datePart;
  }

  // Category -> Tags
  if (jekyllFm.category) {
    docusaurusFm.tags = `[${jekyllFm.category}]`;
  } else if (jekyllFm.categories) {
    docusaurusFm.tags = `[${jekyllFm.categories}]`;
  }

  // Authors
  docusaurusFm.authors = '[mahout-team]';

  return docusaurusFm;
}

/**
 * Generate Docusaurus frontmatter string
 */
function generateFrontmatter(frontmatter) {
  const lines = ['---'];
  for (const [key, value] of Object.entries(frontmatter)) {
    lines.push(`${key}: ${value}`);
  }
  lines.push('---\n');
  return lines.join('\n');
}

/**
 * Process a single blog post
 */
function migratePost(srcPath, destPath) {
  const content = fs.readFileSync(srcPath, 'utf-8');
  const filename = path.basename(srcPath);

  const { frontmatter: jekyllFm, body } = parseFrontmatter(content);
  const docusaurusFm = transformFrontmatter(jekyllFm, filename);

  const finalContent = generateFrontmatter(docusaurusFm) + body;

  fs.writeFileSync(destPath, finalContent);
}

/**
 * Create authors.yml file
 */
function createAuthorsFile() {
  const authorsPath = path.join(DEST_DIR, 'authors.yml');
  const content = `mahout-team:
  name: Apache Mahout Team
  url: https://github.com/apache/mahout
  image_url: https://mahout.apache.org/img/mahout-logo-blue.svg
`;
  fs.writeFileSync(authorsPath, content);
  console.log('Created authors.yml');
}

/**
 * Main migration function
 */
function main() {
  console.log('Starting blog post migration...\n');
  console.log(`Source: ${SOURCE_DIR}`);
  console.log(`Destination: ${DEST_DIR}\n`);

  // Ensure destination exists
  if (!fs.existsSync(DEST_DIR)) {
    fs.mkdirSync(DEST_DIR, { recursive: true });
  }

  // Check source exists
  if (!fs.existsSync(SOURCE_DIR)) {
    console.error('Source directory does not exist!');
    process.exit(1);
  }

  // Create authors file
  createAuthorsFile();

  // Get all markdown files
  const files = fs.readdirSync(SOURCE_DIR)
    .filter(f => f.endsWith('.md'));

  console.log(`Found ${files.length} blog posts to migrate.\n`);

  let migrated = 0;
  let skipped = 0;

  for (const file of files) {
    const srcPath = path.join(SOURCE_DIR, file);
    const destPath = path.join(DEST_DIR, file);

    try {
      // Check if it's a file (not directory)
      const stat = fs.statSync(srcPath);
      if (!stat.isFile()) {
        skipped++;
        continue;
      }

      migratePost(srcPath, destPath);
      migrated++;
      console.log(`  Migrated: ${file}`);
    } catch (err) {
      console.error(`  Error migrating ${file}: ${err.message}`);
      skipped++;
    }
  }

  console.log(`\nMigration complete!`);
  console.log(`  Migrated: ${migrated}`);
  console.log(`  Skipped: ${skipped}`);
}

// Run migration
main();
