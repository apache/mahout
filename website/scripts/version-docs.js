#!/usr/bin/env node

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

const { spawnSync } = require('child_process');

function usage() {
  console.error('Usage: npm run version -- <release-version>');
  console.error('');
  console.error('Example: npm run version -- 0.7');
}

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    process.exit(result.status);
  }
}

function main() {
  const version = process.argv[2];

  if (!version || version.startsWith('-')) {
    usage();
    process.exit(1);
  }

  console.log(`Preparing versioned docs for ${version}...`);
  run('npm', ['run', 'sync']);
  run('npm', ['run', 'docusaurus', '--', 'docs:version', version]);
  run('npm', ['run', 'build']);
}

main();
