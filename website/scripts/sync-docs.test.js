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
  titleFromFilename,
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

test('parseFrontmatter handles real YAML, not just key: value lines', () => {
  // Each of these was misparsed by the old split(':') parser (#1308).
  const { frontmatter } = parseFrontmatter(
    [
      '---',
      'title: "Ratio: 3:1 # not a comment"',
      'tags: [release, qumat, qdp]',
      'authors:',
      '  - jiekaichang',
      '  - mahout-team',
      'description: >',
      '  Folded text that',
      '  spans two lines.',
      'sidebar_custom_props:',
      '  badge: new',
      'sidebar_position: 3',
      'custom_edit_url: null',
      '---',
      'body',
    ].join('\n'),
  );
  assert.deepEqual(frontmatter, {
    title: 'Ratio: 3:1 # not a comment',
    tags: ['release', 'qumat', 'qdp'],
    authors: ['jiekaichang', 'mahout-team'],
    description: 'Folded text that spans two lines.\n',
    sidebar_custom_props: { badge: 'new' },
    sidebar_position: 3,
    custom_edit_url: null,
  });
});

test('parseFrontmatter accepts CRLF line endings', () => {
  const { frontmatter, body } = parseFrontmatter(
    '---\r\ntitle: Windows\r\n---\r\nbody\r\n',
  );
  assert.deepEqual(frontmatter, { title: 'Windows' });
  assert.equal(body, 'body\r\n');
});

test('parseFrontmatter treats an empty block as no frontmatter', () => {
  assert.deepEqual(parseFrontmatter('---\n---\nbody\n'), {
    frontmatter: {},
    body: 'body\n',
  });
});

test('processMarkdown reports invalid YAML with the file name', () => {
  assert.throws(
    () => processMarkdown('---\ntitle: [oops\n---\nx\n', 'docs/bad.md'),
    /Invalid YAML frontmatter in docs\/bad\.md/,
  );
});

test('generateFrontmatter round-trips through parseFrontmatter', () => {
  const data = {
    title: "Summary of 'A: B' #1",
    sidebar_label: 'PR Policy & Review',
    date: new Date('2026-02-02T00:00:00Z'),
    tags: ['release', 'qumat'],
    sidebar_position: 2,
    custom_edit_url: null,
    sidebar_custom_props: { badge: 'new' },
    yes_string: 'yes',
    number_string: '0.6',
  };
  const out = generateFrontmatter(data);
  assert.deepEqual(parseFrontmatter(`${out}body`).frontmatter, data);
});

test('generateFrontmatter keeps the existing on-disk style', () => {
  // Matches how the docs are written today, so the synced files diff
  // cleanly against the old script's output.
  assert.equal(
    generateFrontmatter({
      title: "Summary of 'X: Y'",
      date: new Date('2024-03-04T00:00:00Z'),
      tags: ['release', 'qumat', 'qdp'],
      toc_min_heading_level: 2,
    }),
    [
      '---',
      'title: "Summary of \'X: Y\'"',
      'date: 2024-03-04',
      'tags: [release, qumat, qdp]',
      'toc_min_heading_level: 2',
      '---',
      '',
    ].join('\n'),
  );
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

test('titleFromFilename keeps known acronyms (#1308)', () => {
  const cases = {
    'docs/qdp/qdp-api.md': 'QDP API',
    'docs/gpu_faq.md': 'GPU FAQ',
    'docs/cuda-and-rocm-setup.md': 'CUDA And ROCm Setup',
    'docs/advanced/pqc.md': 'PQC',
    'docs/qml-with-pytorch.md': 'QML With PyTorch',
    'docs/llms-for-quantum.md': 'LLMs For Quantum',
    'docs/pr-policy-and-review-guidelines.md':
      'PR Policy And Review Guidelines',
    'docs/qumat/getting-started.md': 'Getting Started',
    'docs/papers/Quantum-Kernel-Estimation.md': 'Quantum Kernel Estimation',
    'docs/adr/001-sync-docs.md': 'Sync Docs',
  };
  for (const [file, title] of Object.entries(cases)) {
    assert.equal(titleFromFilename(file), title, file);
  }
});

test('titleFromFilename does not split words that contain an acronym', () => {
  // Only whole words are matched: "apis" is mapped, "rapid" is not.
  assert.equal(titleFromFilename('rapid-apis.md'), 'Rapid APIs');
});

test('transformFrontmatter never overwrites an explicit title', () => {
  for (const title of ['qdp api', '', 0, 2024]) {
    assert.deepEqual(
      transformFrontmatter({ title }, 'docs/qdp-api.md'),
      { title },
      JSON.stringify(title),
    );
  }
});

test('transformFrontmatter derives a title only when one is missing', () => {
  assert.deepEqual(transformFrontmatter({}, 'docs/qdp/gpu-faq.md'), {
    title: 'GPU FAQ',
  });
  assert.deepEqual(
    transformFrontmatter({ title: null }, 'docs/qdp/gpu-faq.md'),
    { title: 'GPU FAQ' },
  );
});

test('transformFrontmatter leaves the title to a leading H1', () => {
  // Docusaurus already uses the first `# Heading` as the title, which is
  // better than anything we can guess from the file name.
  for (const body of ['# QDP: GPU FAQ\n', '\nQDP FAQ\n=======\n']) {
    assert.deepEqual(
      transformFrontmatter({}, 'docs/qdp/gpu-faq.md', body),
      {},
      JSON.stringify(body),
    );
  }
  assert.deepEqual(
    transformFrontmatter({}, 'docs/qdp/gpu-faq.md', '## Not a title\n'),
    { title: 'GPU FAQ' },
  );
});

test('transformFrontmatter does not mutate its input', () => {
  const input = { layout: 'page' };
  transformFrontmatter(input, 'docs/x.md');
  assert.deepEqual(input, { layout: 'page' });
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

test('processMarkdown passes every link form through unchanged', () => {
  // The old regex-based transformLinks() only saw inline [text](url)
  // links and missed the forms below; the body is now never rewritten.
  const body = [
    'Inline [a](./a.md), [b](../b/index.md#x), [c](https://x.org/c.md).',
    'Autolink <https://mahout.apache.org/docs/>.',
    'Reference [guide][g] and [g].',
    '',
    '[g]: ./guide.md "Guide"',
    '',
    '<a href="./html.md">HTML link</a>',
    '<Link to="./jsx.md">JSX link</Link>',
    '![img](./assets/pic.png)',
    '```md',
    '[not a link](./code.md)',
    '```',
    '',
  ].join('\n');
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
