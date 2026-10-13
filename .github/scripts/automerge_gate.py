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
"""Keep GitHub auto-merge from merging a pull request whose CI failed.

This backs the ``Auto-merge gate`` required status check on ``main`` (see
``.github/workflows/automerge-gate.yml`` and ``.asf.yaml``).

Background: GitHub auto-merge only waits for *required* status checks. Making
every CI job required would also block committers from merging manually when
they decide a failure is acceptable (e.g. a flaky external check), which is
not what the project wants. So instead a single required check waits for the
``pull_request`` workflows of the head commit and then:

* while any of them is still running -> exit 1 ("waiting"); the check is
  re-evaluated every time one of them completes (``workflow_run`` trigger);
* if every one of them passed (or was skipped) -> exit 0; auto-merge, if
  enabled, merges the PR;
* if any of them failed -> **disable auto-merge** on the PR (if it was
  enabled) and leave a comment, then exit 0 so that a committer can still
  merge manually. The failed CI job stays visible on the PR.

Urgent changes do not go through this script at all: PMC members are bypass
actors of the ``main`` ruleset in ``.asf.yaml`` and can merge a pull request
from the GitHub UI without waiting for any required check.

Only the Python standard library is used so the script runs in CI and locally:

    GITHUB_TOKEN=$(gh auth token) DRY_RUN=1 python .github/scripts/automerge_gate.py \\
        --repo apache/mahout --sha <head sha>

``DRY_RUN=1`` performs no mutation (auto-merge is not disabled, no comment).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_URL = os.environ.get("GITHUB_API_URL", "https://api.github.com")
GRAPHQL_URL = os.environ.get("GITHUB_GRAPHQL_URL", "https://api.github.com/graphql")
# Only workflows triggered by these events are CI for the pull request. In
# particular this excludes "PR Approval Check" (pull_request_target), which is
# its own required status check, and this gate itself (workflow_run).
PR_EVENTS = {"pull_request"}
# Conclusions that do not count as a CI failure.
OK_CONCLUSIONS = {"success", "skipped", "neutral"}
# A run that is waiting for a maintainer to approve it (fork PRs) is pending,
# not failed.
PENDING_CONCLUSIONS = {"action_required"}


class GitHubError(RuntimeError):
    """Raised when the GitHub API returns an unexpected response."""


def _request(token: str, url: str, data: dict[str, Any] | None = None) -> Any:
    body = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method="POST" if body is not None else "GET",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "apache-mahout-automerge-gate",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(f"{request.method} {url} -> HTTP {exc.code}: {text}") from exc


def _api(
    token: str, path: str, params: dict[str, str] | None = None, data: Any = None
) -> Any:
    url = f"{API_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _request(token, url, data)


def _graphql(token: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    result = _request(token, GRAPHQL_URL, {"query": query, "variables": variables})
    if result.get("errors"):
        raise GitHubError(f"GraphQL: {json.dumps(result['errors'])}")
    return result["data"]


def workflow_runs_for_sha(token: str, repo: str, sha: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    page = 1
    while True:
        chunk = _api(
            token,
            f"/repos/{repo}/actions/runs",
            {"head_sha": sha, "per_page": "100", "page": str(page)},
        )
        runs.extend(chunk.get("workflow_runs") or [])
        if len(chunk.get("workflow_runs") or []) < 100:
            return runs
        page += 1


def latest_pr_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map workflow name -> its most recent ``pull_request`` run for the commit.

    A re-run reuses the run id (``run_attempt`` increases) and the API reports
    the latest attempt, so keeping the newest run per workflow is enough.
    """
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.get("event") not in PR_EVENTS:
            continue
        name = run.get("name") or str(run.get("workflow_id"))
        if name not in latest or run["id"] > latest[name]["id"]:
            latest[name] = run
    return latest


def classify(run: dict[str, Any]) -> str:
    """Return ``"pending"``, ``"ok"`` or ``"failed"`` for a workflow run."""
    if run.get("status") != "completed":
        return "pending"
    conclusion = run.get("conclusion") or ""
    if conclusion in PENDING_CONCLUSIONS:
        return "pending"
    if conclusion in OK_CONCLUSIONS:
        return "ok"
    return "failed"


def open_prs_with_head(token: str, repo: str, sha: str) -> list[dict[str, Any]]:
    prs = _api(token, f"/repos/{repo}/commits/{sha}/pulls", {"per_page": "100"})
    # Only act on PRs whose *current* head is this commit: a failure on an
    # already superseded commit must not touch the PR.
    return [
        pr
        for pr in prs
        if pr.get("state") == "open" and (pr.get("head") or {}).get("sha") == sha
    ]


def auto_merge_state(token: str, repo: str, number: int) -> tuple[str, str | None]:
    """Return ``(node_id, enabled_by_login_or_None)`` for a pull request."""
    owner, name = repo.split("/", 1)
    data = _graphql(
        token,
        """
        query($owner: String!, $name: String!, $number: Int!) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $number) {
              id
              autoMergeRequest { enabledBy { login } }
            }
          }
        }
        """,
        {"owner": owner, "name": name, "number": number},
    )
    pr = data["repository"]["pullRequest"]
    request = pr.get("autoMergeRequest")
    enabled_by = ((request or {}).get("enabledBy") or {}).get("login")
    return pr["id"], (enabled_by or "(unknown)") if request else None


def disable_auto_merge(token: str, node_id: str) -> None:
    _graphql(
        token,
        """
        mutation($id: ID!) {
          disablePullRequestAutoMerge(input: {pullRequestId: $id}) {
            pullRequest { number }
          }
        }
        """,
        {"id": node_id},
    )


def comment(token: str, repo: str, number: int, body: str) -> None:
    _api(token, f"/repos/{repo}/issues/{number}/comments", data={"body": body})


def failure_comment(sha: str, failed: list[dict[str, Any]], enabled_by: str) -> str:
    lines = [
        f"**Auto-merge disabled** (it was enabled by @{enabled_by}): "
        f"CI failed for {sha[:7]}.",
        "",
    ]
    lines.extend(
        f"- {run['name']}: {run.get('conclusion')} ({run.get('html_url')})"
        for run in failed
    )
    lines.extend(
        [
            "",
            "Push a fix and enable auto-merge again, or a committer can merge "
            "manually if the failure is acceptable. Auto-merge is never "
            "re-enabled automatically.",
            "",
            "_Posted by the `Auto-merge gate` workflow "
            "(`.github/workflows/automerge-gate.yml`)._",
        ]
    )
    return "\n".join(lines)


def gate(token: str, repo: str, sha: str, dry_run: bool) -> tuple[bool, list[str]]:
    """Evaluate the gate. Returns ``(ok, log_lines)``."""
    lines: list[str] = []
    runs = latest_pr_runs(workflow_runs_for_sha(token, repo, sha))
    if not runs:
        # Cannot happen when triggered by workflow_run, but be explicit.
        lines.append(f"No pull_request workflow runs found for {sha[:7]}.")
        return False, lines

    pending: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for name in sorted(runs):
        run = runs[name]
        state = classify(run)
        lines.append(
            f"{name}: {run.get('status')}/{run.get('conclusion') or '-'} -> {state}"
        )
        if state == "pending":
            pending.append(run)
        elif state == "failed":
            failed.append(run)

    if pending:
        names = ", ".join(run["name"] for run in pending)
        lines.append(f"Waiting for: {names}. Re-evaluated when they complete.")
        return False, lines

    if not failed:
        lines.append("All pull_request workflows passed or were skipped.")
        return True, lines

    names = ", ".join(run["name"] for run in failed)
    lines.append(f"CI failed: {names}.")
    for pr in open_prs_with_head(token, repo, sha):
        number = int(pr["number"])
        node_id, enabled_by = auto_merge_state(token, repo, number)
        if enabled_by is None:
            lines.append(f"PR #{number}: auto-merge not enabled; nothing to do.")
            continue
        if dry_run:
            lines.append(
                f"PR #{number}: auto-merge enabled by @{enabled_by}; "
                "DRY_RUN set, would disable it and comment."
            )
            continue
        disable_auto_merge(token, node_id)
        comment(token, repo, number, failure_comment(sha, failed, enabled_by))
        lines.append(
            f"PR #{number}: auto-merge (enabled by @{enabled_by}) disabled; "
            "comment posted."
        )
    lines.append(
        "Passing the gate so that a committer can still merge manually; "
        "the failed job remains visible on the PR."
    )
    return True, lines


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
        "--sha",
        default=os.environ.get("HEAD_SHA"),
        help="pull request head commit (default: $HEAD_SHA)",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token or not args.repo or not args.sha:
        parser.error(
            "GITHUB_TOKEN, --repo/$GITHUB_REPOSITORY and --sha/$HEAD_SHA are required"
        )
    dry_run = os.environ.get("DRY_RUN", "") not in {"", "0", "false"}

    try:
        ok, lines = gate(token, args.repo, args.sha, dry_run)
    except GitHubError as exc:
        print(f"::error::{exc}")
        return 1

    verdict = (
        "PASS" if ok else "WAITING: not all pull_request workflows have completed."
    )
    lines.append(verdict)
    print("\n".join(lines))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("### Auto-merge gate\n\n")
            summary.writelines(f"- {line}\n" for line in lines)
    if not ok:
        print(f"::notice::{verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
