#!/bin/bash
# Generate RC Testing Issue for Apache Mahout
# Usage: ./generate-rc-issue.sh <qumat_version> <qdp_version> [milestone_name]
# Example: ./generate-rc-issue.sh 0.5.0rc1 0.1.0rc1 "Qumat 0.5.0"

QUMAT_VERSION=${1:-"0.5.0rc1"}
QDP_VERSION=${2:-"0.1.0rc1"}
MILESTONE=${3:-"Qumat 0.5.0"}

cat << EOF
# Testing Apache Mahout Qumat ${QUMAT_VERSION}
We kindly request all contributors to help test the Release Candidate for [qumat ${QUMAT_VERSION}](https://pypi.org/project/qumat/${QUMAT_VERSION}/) and [qumat-qdp ${QDP_VERSION}](https://pypi.org/project/qumat-qdp/${QDP_VERSION}/).

## Requirements

- Python 3.10, 3.11, or 3.12
- **For QDP extension**: Linux with NVIDIA GPU and CUDA

## Installation

\`\`\`bash
# Core package
pip install --pre "qumat==${QUMAT_VERSION}"

# With QDP extension
pip install --pre "qumat[qdp]==${QUMAT_VERSION}"
\`\`\`

## Changes to Test

EOF

# Get merged PRs from milestone with PR URL
gh pr list --repo apache/mahout --state merged --search "milestone:\"${MILESTONE}\"" --limit 500 --json number,title,author \
  --jq '.[] | "- [ ] [\(.title) (#\(.number))](https://github.com/apache/mahout/pull/\(.number)): @\(.author.login)"' 2>/dev/null

cat << EOF

## Compatibility

- [ ] Python 3.10
- [ ] Python 3.11
- [ ] Python 3.12

EOF

echo -n "Thanks to all who contributed to the release: "
gh pr list --repo apache/mahout --state merged --search "milestone:\"${MILESTONE}\"" --limit 500 --json author \
  --jq '.[].author.login' 2>/dev/null | sort -u | sed 's/^/@/' | tr '\n' ' '

cat << EOF


### Committer

- [ ] I acknowledge that I am a maintainer/committer of the Apache Mahout project.
EOF
