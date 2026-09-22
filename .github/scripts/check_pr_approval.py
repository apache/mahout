#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Check that a pull request has at least one approving review from a committer.

This backs the ``Committer approval`` required status check on ``main``
(see ``.github/workflows/pr-approval-check.yml`` and ``.asf.yaml``).

Rules:

* A pull request opened by a PMC member (``.github/pmc-members.txt``) is exempt.
* Otherwise, at least one reviewer whose *latest* review is ``APPROVED`` must have
  write access to the repository. Per ASF policy every committer has write access,
  so "has write access" is used as the definition of "is a committer".
* Following GitHub's own semantics, a ``COMMENTED`` review does not revoke an
  earlier approval, while ``CHANGES_REQUESTED`` or a dismissal does.

Only the Python standard library is used so the script runs in CI and locally:

    GITHUB_TOKEN=$(gh auth token) python .github/scripts/check_pr_approval.py \\
        --repo apache/mahout --pr 1234

Exit status is 0 when the PR may be merged and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
DEFAULT_PMC_FILE = Path(__file__).resolve().parents[1] / "pmc-members.txt"
# Review states that change a reviewer's standing. COMMENTED is deliberately
# excluded: on GitHub a comment-only review leaves a previous approval in place.
DECISIVE_STATES = {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}


class GitHubError(RuntimeError):
    """Raised when the GitHub API returns an unexpected response."""


def _api(token: str, path: str, params: dict[str, str] | None = None) -> Any:
    url = f"{API_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "apache-mahout-pr-approval-check",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(f"GET {path} -> HTTP {exc.code}: {body}") from exc


def _paginate(token: str, path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        chunk = _api(token, path, {"per_page": "100", "page": str(page)})
        items.extend(chunk)
        if len(chunk) < 100:
            return items
        page += 1


def load_pmc_members(path: Path) -> set[str]:
    """Return the lower-cased GitHub usernames listed in ``path``.

    Blank lines and ``#`` comments (full-line or trailing) are ignored.
    """
    members: set[str] = set()
    if not path.is_file():
        return members
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            members.add(line.lower())
    return members


def latest_decisive_reviews(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map each reviewer login (lower-cased key) to their latest decisive review."""
    latest: dict[str, dict[str, Any]] = {}
    # The API returns reviews in chronological order; sort defensively anyway.
    for review in sorted(reviews, key=lambda r: (r.get("submitted_at") or "", r["id"])):
        if review.get("state") not in DECISIVE_STATES:
            continue
        user = review.get("user") or {}
        login = user.get("login")
        if not login or user.get("type") == "Bot":
            continue
        latest[login.lower()] = review
    return latest


def has_write_access(token: str, repo: str, login: str) -> bool:
    path = f"/repos/{repo}/collaborators/{urllib.parse.quote(login)}/permission"
    data = _api(token, path)
    permissions = (data.get("user") or {}).get("permissions") or {}
    if "push" in permissions:
        return bool(permissions["push"])
    return data.get("permission") in {"write", "admin"}


def check(
    token: str, repo: str, pr_number: int, pmc_members: set[str]
) -> tuple[bool, list[str]]:
    """Evaluate the rule. Returns ``(ok, log_lines)``."""
    lines: list[str] = []
    pr = _api(token, f"/repos/{repo}/pulls/{pr_number}")
    author = (pr.get("user") or {}).get("login", "")
    lines.append(
        f"PR #{pr_number} by @{author} -> {pr['base']['ref']} "
        f"(head {pr['head']['sha'][:7]})"
    )

    if author.lower() in pmc_members:
        lines.append(
            f"@{author} is a PMC member: exempt from the committer-approval rule."
        )
        return True, lines

    reviews = _paginate(token, f"/repos/{repo}/pulls/{pr_number}/reviews")
    approvers = [
        review["user"]["login"]
        for login, review in latest_decisive_reviews(reviews).items()
        if review["state"] == "APPROVED" and login != author.lower()
    ]
    if not approvers:
        lines.append("No approving reviews found.")
        return False, lines

    committer_approvers: list[str] = []
    for login in sorted(approvers, key=str.lower):
        if has_write_access(token, repo, login):
            committer_approvers.append(login)
            lines.append(f"Approved by @{login} (committer).")
        else:
            lines.append(
                f"Approved by @{login} (not a committer; does not satisfy the rule)."
            )

    if committer_approvers:
        return True, lines
    lines.append("No approving review from a committer.")
    return False, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=os.environ.get("PR_NUMBER"),
        help="pull request number (default: $PR_NUMBER)",
    )
    parser.add_argument(
        "--pmc-file",
        type=Path,
        default=DEFAULT_PMC_FILE,
        help=f"PMC member list (default: {DEFAULT_PMC_FILE})",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token or not args.repo or args.pr is None:
        parser.error(
            "GITHUB_TOKEN, --repo/$GITHUB_REPOSITORY and --pr/$PR_NUMBER are required"
        )

    pmc_members = load_pmc_members(args.pmc_file)
    if not pmc_members:
        print(
            f"::warning::No PMC members loaded from {args.pmc_file}; "
            "no author is exempt."
        )

    try:
        ok, lines = check(token, args.repo, int(args.pr), pmc_members)
    except GitHubError as exc:
        print(f"::error::{exc}")
        return 1

    if ok:
        verdict = "PASS: at least one committer approval (or PMC author)."
    else:
        verdict = "FAIL: needs at least one approving review from a committer."
    lines.append(verdict)
    print("\n".join(lines))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("### Committer approval\n\n")
            summary.writelines(f"- {line}\n" for line in lines)
    if not ok:
        print(
            f"::error::{verdict} See docs/community/pr-policy-and-review-guidelines.md."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
