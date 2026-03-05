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

---
title: PR Policy and Review Guidelines
sidebar_label: PR Policy & Review
---

# PR Policy and Review Guidelines

This document defines Apache Mahout's pull request and review process, aligned with ASF principles of openness, consensus, and merit.

## Scope

These rules apply to all pull requests in `apache/mahout`, including code, tests, CI, and documentation.

## Core Principles

- Discuss significant changes in public (GitHub issue and/or `dev@mahout.apache.org`).
- Keep decisions transparent and archived in project systems.
- Prefer small, reviewable PRs over large multi-purpose changes.
- Resolve technical disagreement through evidence, tests, and community discussion.

## Contributor Requirements

Before opening a PR:

- Link to an issue when applicable.
- Follow the PR template in [`PULL_REQUEST_TEMPLATE`](https://github.com/apache/mahout/blob/main/.github/PULL_REQUEST_TEMPLATE).
- Include tests for behavior changes.
- Update docs for user-visible changes.
- Ensure CI and required checks pass.

For non-trivial code contributions, contributors are expected to have an ASF ICLA on file.

## PR Author Guidelines

- Keep PRs focused on one concern (bug fix, feature slice, refactor, docs, etc.).
- Describe `Why` and `How` clearly in the PR description.
- Mark breaking or risky changes explicitly.
- Respond to review feedback promptly and keep discussion on the PR thread.
- Do not force-push in ways that hide review context after substantial review has started.

## Review Expectations

### General

- Anyone in the community can review and comment.
- CODEOWNERS define suggested reviewers, not exclusive authority.
- Reviewers should focus on correctness, API/UX impact, test coverage, security, and maintainability.

### For Contributors and Reviewers

- Treat review comments as collaborative feedback.
- Resolve open questions with tests, benchmarks, or design rationale when possible.
- Keep review discussion on the PR thread so decisions are public and traceable.

### Committer Merge Criteria (Committers Only)

When deciding whether to merge, committers should verify all of the following:

- Required CI checks are green.
- At least one committer approves.
- All blocking comments are resolved.

The following changes should receive at least two committer approvals and prior public discussion (issue or `dev@` thread):

- API or behavior-breaking changes.
- Architecture or major dependency changes.
- Security-sensitive changes.
- Release/build/reproducibility process changes.

## Blocking and Objections

- A blocking objection must include concrete technical reasons and, when possible, a suggested path to resolution.
- PRs with unresolved blocking objections must not be merged.
- If consensus is unclear, escalate discussion to `dev@mahout.apache.org`.

## Security and Legal Checks

Before merge, confirm:

- No secrets, credentials, or private data are committed.
- New dependencies and copied code are license-compatible with Apache-2.0.
- License/notice updates are included when required (`LICENSE`, `NOTICE`).

## Recommended SLA (Non-binding)

- First review response target: within 5 business days.
- Author follow-up target after requested changes: within 7 business days.

These are goals, not strict requirements.

## References

- ASF Contributor Guide: https://community.apache.org/contributors/
- ASF Committer Guide: https://community.apache.org/committers/
- ASF Voting Process: https://www.apache.org/foundation/voting.html
- ASF License and Contribution FAQs: https://www.apache.org/licenses/
- Project contribution guide: [`CONTRIBUTING.md`](https://github.com/apache/mahout/blob/main/CONTRIBUTING.md)
- CODEOWNERS: [`.github/CODEOWNERS`](https://github.com/apache/mahout/blob/main/.github/CODEOWNERS)
