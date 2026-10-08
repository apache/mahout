/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Unit tests for scripts/sync-docs.js. Uses only Node's built-in test
// runner, so no extra test framework is needed. Run with `npm test`.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  shouldExclude,
  parseFrontmatter,
  generateFrontmatter,
  transformFrontmatter,
  processMarkdown,
  syncDirectory,
  cleanDestination,
} = require('./sync-docs.js');

test('shouldExclude skips hidden files, blog, api and Python caches', () => {
  for (const name of ['.DS_Store', 'blog', 'api', '__pycache__', 'x.pyc']) {
    assert.equal(shouldExclude(name), true, name);
  }
  for (const name of ['index.md', 'qdp', 'blog.md', 'apis']) {
    assert.equal(shouldExclude(name), false, name);
  }
});

test('parseFrontmatter returns empty data when there is no frontmatter', () => {
  const content = '# Title\n\nBody\n';
  assert.deepEqual(parseFrontmatter(content), {
    frontmatter: {},
    body: content,
  });
});

test('parseFrontmatter splits simple key/value frontmatter from body', () => {
  const { frontmatter, body } = parseFrontmatter(
    '---\ntitle: Hello\nsidebar_label: Hi\n---\n\n# Hello\n',
  );
  assert.equal(frontmatter.title, 'Hello');
  assert.equal(frontmatter.sidebar_label, 'Hi');
  assert.equal(body, '\n# Hello\n');
});

test('parseFrontmatter strips matching quotes', () => {
  const { frontmatter } = parseFrontmatter(
    '---\ntitle: "Summary of \'A: B\'"\n---\nx\n',
  );
  assert.equal(frontmatter.title, "Summary of 'A: B'");
});

test('generateFrontmatter quotes values containing a colon', () => {
  assert.equal(
    generateFrontmatter({ title: 'A: B', sidebar_label: 'B' }),
    '---\ntitle: "A: B"\nsidebar_label: B\n---\n',
  );
});

test('transformFrontmatter drops Jekyll-only fields', () => {
  const out = transformFrontmatter(
    { layout: 'page', permalink: '/x', title: 'X' },
    'docs/x.md',
  );
  assert.deepEqual(out, { title: 'X' });
});

test('transformFrontmatter does not add a title to index pages', () => {
  assert.deepEqual(transformFrontmatter({}, 'docs/qdp/index.md'), {});
});

test('processMarkdown leaves relative .md links untouched (#1301, #1305)', () => {
  // Docusaurus resolves `./page.md` links to version-aware URLs itself,
  // so the sync step must hand them over unchanged.
  const body =
    'See [PR policy](./pr-policy-and-review-guidelines.md) and ' +
    '[QDP](../qdp/index.md#install).\n';
  const out = processMarkdown(`---\ntitle: T\n---\n${body}`, 'docs/t.md');
  assert.equal(out, `---\ntitle: T\n---\n${body}`);
});

test('processMarkdown removes frontmatter left empty after transforms', () => {
  const out = processMarkdown('---\nlayout: page\n---\n# Home\n', 'index.md');
  assert.equal(out, '# Home\n');
});

function makeTempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sync-docs-test-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test('syncDirectory copies markdown (transformed) and assets', (t) => {
  const root = makeTempDir(t);
  const src = path.join(root, 'src');
  const dest = path.join(root, 'dest');
  fs.mkdirSync(path.join(src, 'qdp'), { recursive: true });
  fs.mkdirSync(path.join(src, 'blog'));
  fs.writeFileSync(
    path.join(src, 'qdp', 'guide.md'),
    '---\nlayout: page\ntitle: Guide\n---\nBody\n',
  );
  fs.writeFileSync(path.join(src, 'qdp', 'diagram.png'), 'PNG');
  fs.writeFileSync(path.join(src, 'blog', 'post.md'), 'skipped');
  fs.writeFileSync(path.join(src, '.hidden'), 'skipped');

  const stats = syncDirectory(src, dest);

  assert.deepEqual(stats, { files: 2, dirs: 1 });
  assert.equal(
    fs.readFileSync(path.join(dest, 'qdp', 'guide.md'), 'utf-8'),
    '---\ntitle: Guide\n---\nBody\n',
  );
  assert.equal(
    fs.readFileSync(path.join(dest, 'qdp', 'diagram.png'), 'utf-8'),
    'PNG',
  );
  assert.equal(fs.existsSync(path.join(dest, 'blog')), false);
  assert.equal(fs.existsSync(path.join(dest, '.hidden')), false);
});

test('cleanDestination keeps .gitignore and removes everything else', (t) => {
  const dest = makeTempDir(t);
  fs.writeFileSync(path.join(dest, '.gitignore'), '*\n');
  fs.writeFileSync(path.join(dest, 'stale.md'), 'old');
  fs.mkdirSync(path.join(dest, 'old-dir'));

  const log = console.log;
  console.log = () => {};
  try {
    cleanDestination(dest);
  } finally {
    console.log = log;
  }

  assert.deepEqual(fs.readdirSync(dest), ['.gitignore']);
});
