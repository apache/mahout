<!-- SPDX-License-Identifier: Apache-2.0
     https://www.apache.org/licenses/LICENSE-2.0 -->

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Apache Mahout — project manifest](#apache-mahout--project-manifest)
  - [Identity](#identity)
  - [Repositories](#repositories)
  - [Mailing lists](#mailing-lists)
  - [Tools enabled](#tools-enabled)
  - [CVE tooling](#cve-tooling)
  - [GitHub project board](#github-project-board)
  - [Project chat](#project-chat)
  - [Mail sources](#mail-sources)
    - [Backend declaration](#backend-declaration)
    - [Per-backend config](#per-backend-config)
  - [Issue-template fields](#issue-template-fields)
  - [Security workflow configuration](#security-workflow-configuration)
    - [CVE authority](#cve-authority)
    - [Governance](#governance)
    - [Security inbox](#security-inbox)
    - [Forwarders](#forwarders)
    - [Mail provider](#mail-provider)
    - [Archive system](#archive-system)
    - [Project metadata](#project-metadata)
    - [Security cross-reference](#security-cross-reference)
    - [Tracker](#tracker)
    - [Scope detection](#scope-detection)
    - [Release process](#release-process)
    - [Roster](#roster)
    - [Product](#product)
  - [Pointers to sibling files](#pointers-to-sibling-files)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<!-- SPDX-License-Identifier: Apache-2.0
     https://www.apache.org/licenses/LICENSE-2.0 -->

# Apache Mahout — project manifest

This is the **project configuration** for `mahout`.
Every skill under [`../../.claude/skills/`](../../../skills/)
reads the project name from `<project-config>/project.md` and then loads this manifest to resolve project-specific identity,
repositories, mailing lists, and references to the other files in
this directory.

**Note on Auto-sourcing:** Stable fields under Identity and Repositories
(`upstream_repo`, `upstream_default_branch`, `product_family_url`, labels)
can be automatically derived from GitHub repository metadata via
`gh repo view` during `/magpie-setup adopt` or `/magpie-setup upgrade`, for
any adopter. The **Mailing lists** are auto-sourced **only for
`organization: ASF`** projects (from `.asf.yaml`, with `*.apache.org`
defaults); a non-ASF `organization` fills them in by hand. Hand-editing is
only required when a source is absent, incomplete, or you wish to override
a derived value.

Grep for `TODO` to see every field you still need to fill in:

```bash
grep -n TODO projects/<name>/project.md
```

## Identity

| Key | Value |
|---|---|
| `organization` | `ASF` |
| `project_name` | Apache Mahout |
| `vendor` | Apache Software Foundation |
| `short_name` | Mahout |
| `product_family_url` | https://mahout.apache.org/ |

The `vendor` / `project_name` pair is what lands in the `vendor` and
`product` fields of the CVE 5.x record the CVE-JSON generator
produces.

## Repositories

| Key | Value | Purpose |
|---|---|---|
| `tracker_repo` | TODO: e.g. `foo-s/foo-s` | Private security tracker (this repo) |
| `tracker_repo_url` | TODO | |
| `tracker_default_branch` | TODO: e.g. `main` | Default PR target for the tracker repo |
| `tracker_project_board_url` | TODO: URL of the GitHub Project V2 board, if any | Security board |
| `upstream_repo` | `apache/mahout` | Public codebase where fixes land |
| `upstream_repo_url` | https://github.com/apache/mahout | |
| `upstream_default_branch` | `main` | Upstream's default branch — what `<default-branch>` resolves to. Distinct from `tracker_default_branch` |
| `upstream_agents_md_url` | TODO: `https://github.com/<upstream>/blob/main/AGENTS.md` | Conventions this repo mirrors |
| `upstream_contributing_docs_url` | https://github.com/apache/mahout/blob/main/CONTRIBUTING.md | |
| `upstream_genai_disclosure_anchor` | TODO: URL + anchor for the project's Gen-AI disclosure guideline | |
| `upstream_security_policy_url` | TODO: `https://github.com/<upstream>/security/policy` | |
| `security_pages_url` | TODO: e.g. `https://foo.apache.org/security` | Public security pages on the project website — updated by the release manager after each advisory (ASF security-committers policy step); read by `security-issue-sync` for the security-pages checklist reminder. Leave unset if the project has no security pages — the skill then uses the plain phrase without a link |

## Mailing lists

*(For `organization: ASF` projects only: auto-sourced from the `.asf.yaml`
`notifications:` block if present, else `*.apache.org` defaults. A non-ASF
`organization` has no `.asf.yaml` — fill these in by hand.)*

| Key | Value | Notes |
|---|---|---|
| `security_list` | `security@mahout.apache.org` | Inbound reports; **not** publicly archived (auto-sourced default) |
| `private_list` | `private@mahout.apache.org` | PMC escalation; **not** publicly archived (auto-sourced default) |
| `users_list` | `user@mahout.apache.org` | Public advisories end up here; publicly archived (auto-sourced from `.asf.yaml` or default) |
| `dev_list` | `dev@mahout.apache.org` | Release `[RESULT][VOTE]` threads; publicly archived (auto-sourced from `.asf.yaml` or default) |
| `announce_list` | `announce@apache.org` | Cross-project announcement list; publicly archived |
| `commits_list` | `commits@mahout.apache.org` | Publicly archived (auto-sourced from `.asf.yaml`) |

Only URLs on publicly archived lists may appear in CVE `references[]` as
`vendor-advisory`; see `../../AGENTS.md` and
[`security-model.md`](security-model.md).

The foundation-wide security forwarding address (e.g. `security@apache.org`
for ASF projects) is org-level — inherited from your organization's
`security_inbox.foundation_security_address`. Do not declare it here.

## Tools enabled

| Capability | Tool | Adapter directory | Config knobs declared here |
|---|---|---|---|
| Issue tracking + project board | `github` | [`../../tools/github/`](../../../tools/github/) | `tracker_repo`, `upstream_repo`, `github_project_board_*`, `issue_template_fields` |
| Source control (VCS) | `github` (Git) — replaceable with a non-Git VCS | [`../../tools/github/source-control.md`](../../../tools/github/source-control.md) | `upstream_repo`, `default_branch`; for a non-Git VCS declare the sibling tool + its working-copy URL |
| Inbound email / drafts | `<one or more mail-source backends>` | [`../../tools/mail-source/contract.md`](../../../tools/mail-source/contract.md) (abstract) + per-backend adapter dirs (`tools/gmail/`, `tools/ponymail/`, `tools/mail-source/imap/`, `tools/mail-source/mbox/`, ...) | See [Mail sources](#mail-sources) below — declare each backend's role (primary / preferred-for-`<op>` / fallback / optional) and `mandatory` flag |
| CVE allocation + record mgmt | *org-level* — inherited from `organizations/<org>/organization.md → cve_authority.tool`; for ASF: `vulnogram` ([`tools/cve-tool-vulnogram/`](../../../tools/cve-tool-vulnogram/)), for independent: `mitre-form` | — | override in [CVE authority](#cve-authority) only if this project differs from its org |
| Security cross-ref | `osv` | [`../../tools/osv/`](../../../tools/osv/) | `security_cross_ref.tool`, `security_cross_ref.ecosystem` |
| Project metadata (rosters / people / releases) | *org-level* — inherited from `organizations/<org>/organization.md → project_metadata.kind`; for ASF: `apache-projects` ([`tools/apache-projects/`](../../../tools/apache-projects/)), for independent: `none` | — | override in [Project metadata](#project-metadata) only if this project differs from its org |
| Project chat | TODO: `slack`, `discord`, or `none` | [`../../tools/chat/`](../../../tools/chat/) (abstract) + adapter dirs (`tools/chat-slack/`, `tools/chat-discord/`) | `chat.kind`, `chat.channels`, `chat.guild_id` — see [Project chat](#project-chat) below; read-only, public channels only |
| Release comms | TODO: the backend that carries release announcements — for ASF: `dev_list` / `announce_list` / `users_list`; for GitHub Releases leave blank | — | whichever release-comms keys the org default or per-project override declares |

To replace a tool (e.g. swap GitHub issues for JIRA), declare an
alternate tool in the table above, add a `tools/<name>/` adapter
directory, and make sure the values the generic skills need are still
reachable from this manifest.

## CVE tooling

The CNA tool, allocation URL, record URL template, and allocation gate are
**org-level** — inherited from your organization's `cve_authority` and
`governance` blocks (see
[`organizations/ASF/organization.md`](../../../organizations/ASF/organization.md)
for ASF defaults,
[`organizations/independent/organization.md`](../../../organizations/independent/organization.md)
for the GitHub-native baseline). Override a key in the
[Security workflow configuration → CVE authority](#cve-authority) block only
if this project's CNA setup genuinely differs from its organization.

For **ASF adopters**, fill in the per-project CNA queue fields below — each
ASF project has its own queue slug and org UUID, so these are not org-level:

| Key | Value |
|---|---|
| `asf_org_id` | TODO: project's CNA org UUID — e.g. `f0158376-9dc2-43b6-827c-5f631a4d8d09` |
| `cna_private_owner` | TODO: e.g. `foo` (CNA_private.owner — the project slug inside the ASF CNA queue) |
| `cna_private_projecturl` | TODO: e.g. `https://foo.apache.org/` |
| `cna_private_userslist` | TODO: e.g. `users@foo.apache.org` |

## GitHub project board

If the project uses a Projects V2 board for its security-issue view,
declare the node IDs below. Fetch with the introspection query in
[`../../tools/github/project-board.md`](../../../tools/github/project-board.md#introspection--re-fetch-the-option-ids).
If the project does not run a board, leave the table blank — skills
treat missing board config as *"no board reconciliation"*.

| Key | Value |
|---|---|
| `project_board_url` | TODO |
| `project_board_number` | TODO |
| `project_board_node_id` | TODO |
| `status_field_node_id` | TODO |

**`Status` column → option-ID mapping** (re-fetch if any write
returns `not found`):

| Column | Option ID |
|---|---|
| `Needs triage` | TODO |
| `Assessed` | TODO |
| `CVE allocated` | TODO |
| `PR created` | TODO |
| `PR merged` | TODO |
| `Fix released` | TODO |
| `Announced` | TODO |

## Project chat

The contributor-growth skills read public chat channels for community signals, through the [`tools/chat/`](../../../tools/chat/) contract.
Leave `kind: none` if the project has no public chat, or does not want it read.

```yaml
chat:
  kind: none         # slack | discord | none
  # guild_id: ""     # Discord only, optional: server (guild) ID when the bot joins several servers
  channels: []       # channel names (Discord also accepts IDs) to read; empty = every public channel —
                     # list them explicitly in a workspace shared by several projects
```

## Mail sources

The skills treat every supported mail backend the same way —
through the abstract operations defined in
[`../../tools/mail-source/contract.md`](../../../tools/mail-source/contract.md).
The adopter declares which backends are configured, what *role*
each plays, and whether any are *mandatory*. The skill's resolution
rule (see the contract) then picks the right backend per operation
at run time.

### Backend declaration

One row per configured backend. **Exactly one** row carries
`role: primary`. Multiple rows may carry `preferred for <op>` to
override the primary for specific operations. `fallback` rows are
tried in order when no preferred / primary backend supports the op.
`mandatory: yes` means the skill **refuses to run** when that
backend is unavailable; `no` means the skill continues with the
remaining backends (and skips ops that no available backend supports).

| Backend | Role | Mandatory | Notes |
|---|---|---|---|
| TODO: `gmail` | TODO: e.g. `primary` | TODO: `yes` / `no` | TODO: e.g. "Triager Gmail account subscribed to `<security-list>` and `<private-list>`" |
| TODO: `ponymail` | TODO: e.g. `fallback` or `preferred for thread_url` | TODO: `yes` / `no` | TODO: e.g. "Read-only archive backstop; install per [`tools/ponymail/tool.md`](../../../tools/ponymail/tool.md)" |
| TODO: *(add more rows as needed — `imap`, `mbox`, `mailman3`, project-specific adapter)* | | | |

> **Mail backend selection is org-level.** The `mail_provider` block in
> your organization manifest sets the primary and fallback backends
> (for ASF: `primary: gmail-mcp`, `fallback: ponymail`; for independent:
> `primary: none`, `fallback: none`). Only declare backends here that
> override or extend what the organization manifest provides. For ASF
> projects where PonyMail is the inherited fallback, declare it here only
> to change its `mandatory` flag or role for this specific project; the
> backend itself is already wired by the organization. Install PonyMail per
> [`../../tools/ponymail/tool.md`](../../../tools/ponymail/tool.md#keeping-the-checkout-current).

Reference adapter docs:
[`tools/gmail/tool.md`](../../../tools/gmail/tool.md) (full read+write),
[`tools/ponymail/tool.md`](../../../tools/ponymail/tool.md) (read-only ASF archive),
[`tools/mail-source/imap/README.md`](../../../tools/mail-source/imap/README.md) (stub),
[`tools/mail-source/mbox/README.md`](../../../tools/mail-source/mbox/README.md) (read-only offline archive — stub),
[`tools/mail-source/mailman3/README.md`](../../../tools/mail-source/mailman3/README.md) (read-only public Mailman 3 / Hyperkitty archive).

### Per-backend config

Per-backend values the generic recipes substitute in. Only fill in
the rows for backends declared above; leave the rest blank or
remove the row.

| Key | Backend | Value |
|---|---|---|
| `security_list_domain` | `gmail` | TODO: e.g. `security.foo.apache.org` — Gmail `list:` operator uses the domain form |
| `ponymail_private_search_url_template` | `ponymail` | TODO |
| `ponymail_public_search_url_template` | `ponymail` | TODO |
| `ponymail_api_url_template` | `ponymail` | TODO |
| `ponymail_thread_url_template` | `ponymail` | `https://lists.apache.org/thread/<hash>?<list>` |
| `imap_host` | `imap` | TODO: e.g. `imap.example.org` |
| `imap_account` | `imap` | TODO: e.g. `security-triage@example.org` |
| `imap_security_list_folder` | `imap` | TODO: e.g. `INBOX.security-list` |
| `imap_drafts_folder` | `imap` | TODO: e.g. `Drafts` (or leave blank to declare `create_draft` unsupported on this adapter) |
| `mbox_archive_path` | `mbox` | TODO: e.g. `/srv/audit/security-list-2024.mbox` |
| `mailman3_archive_url` | `mailman3` | TODO: e.g. `https://mail.example.org/archives` (the Hyperkitty root, without `/api`) |

## Issue-template fields

The skills' body-field roles map to the following concrete `###`
headings in the project's issue template (the concrete YAML file lives in the
adopter's `<upstream>` repo; the generic role → field contract is in The generic role → GitHub-field
contract lives in
[`../../tools/github/issue-template.md`](../../../tools/github/issue-template.md);
the concrete names below are what skills read and write for this
project.

| Role (generic) | Field name | Template type | Required? |
|---|---|---|---|
| `issue-description` | TODO | `textarea` | TODO |
| `public-summary` | TODO | `textarea` | TODO |
| `affected-versions` | TODO | `input` | TODO |
| `security-thread` | TODO | `input` | TODO |
| `public-advisory-url` | TODO | `input` | TODO |
| `reporter-credit` | TODO | `input` | TODO |
| `pr-with-fix` | TODO | `input` | TODO |
| `cwe` | TODO | `input` | TODO |
| `severity` | TODO | `dropdown` | TODO |
| `cve-tool-link` | TODO | `input` | TODO |

## Security workflow configuration

Skills resolve every workflow knob from the three-layer chain
`project.md → organizations/<org>/organization.md → framework default`
(see [`AGENTS.md` § Configuration resolution order](../../../AGENTS.md#configuration-resolution-order)).
The **organization** you named under *Identity* supplies the org-wide
defaults — the CNA tool, the governance gate, the mail / forwarder /
archive backends, the project-metadata source, and the release-manager
lookup cascade. For an ASF project these resolve to the Vulnogram /
PonyMail / `apache-projects-mcp` / ASF-security values in
[`organizations/ASF/organization.md`](../../../organizations/ASF/organization.md);
`organization: independent` inherits the GitHub-native baseline in
[`organizations/independent/organization.md`](../../../organizations/independent/organization.md).

**Declare below only what is specific to this project.** Each block that
is purely org-level says so and shows nothing — copy the matching key
from the organization manifest down into that block only to override it
(the project value wins). The blocks that every adopter fills in keep
their per-project keys.

### CVE authority

Org-level — inherited from your organization's `cve_authority` block
(CNA tool, allocate / record / source-tab URLs, state mapping,
propagation, allocation-email flag, reviewer channel). Override a single
key here only if this project's CNA setup differs from its organization.

### Governance

Org-level except the escalation contact: the allocation gate, gate
label, release-vote gating, private governance list, and roster URL are
inherited. Declare this project's escalation contact:

```yaml
governance:
  # GitHub handle (or external contact) the skills cc / @-mention when
  # escalating beyond the security team.
  # Consumed by: security-issue-sync, pr-management-triage.
  escalation_contact: "@<escalation-contact>"
```

### Security inbox

Org-level except the concrete address: the inbox `kind`, the foundation
security address, the `has_forwarder_relay` flag, the list filter, and
the reporter-response timeout are inherited. Declare this project's
inbound address:

```yaml
security_inbox:
  # The concrete inbound address / channel ID / form URL for this project.
  # Consumed by: security-issue-import, security-issue-sync, canned-responses.
  address: <security-list>
```

**Optional override — `reporter_response_timeout_days`.** Days of
reporter silence, counted from the security team's most recent outbound
message on the reporter thread, before `security-issue-sync` proposes
proceeding with the fix and announcement without further reporter
sign-off. Per
[ASF policy](https://www.apache.org/security/committers.html), an
unresponsive reporter must not block the project team, particularly for
a high-severity or high-impact issue. Your organization declares the
default (14 days for both `ASF` and `independent`); add the key here only
to depart from it — a project handling especially sensitive reports may
want a longer window, one under heavy report volume a shorter one.
Consumed by `security-issue-sync` (Step 1c staleness check / Step 2b
proposal).

```yaml
security_inbox:
  reporter_response_timeout_days: 21   # only if your project differs from its organization
```

### Forwarders

Org-level — inherited from your organization's `forwarders` block (which
relay adapters are enabled and their per-adapter detect / credit rules).
Override here only to enable an extra relay this project uses.

### Mail provider

Org-level — inherited from your organization's `mail_provider` block
(primary + fallback mail backends). Override here only if this project
reads mail from a different backend than its organization.

### Archive system

Org-level — inherited from your organization's `archive_system` block
(public-archive backend + URL templates). Override here only if this
project's advisories surface on a different archive than its organization.

### Project metadata

Org-level — inherited from your organization's `project_metadata` block
(roster / people / releases backend + whether it is mandatory). Override
here only if this project uses a different metadata source.

### Security cross-reference

Vulnerability cross-referencing and alias lookup configuration (OSV.dev / NVD).
Declare this project's default package ecosystem and tool adapter:

```yaml
security_cross_ref:
  # Tool adapter: osv (tools/osv/) | none
  tool: osv
  # Default package ecosystem for vulnerability queries (e.g. PyPI, Maven, npm, Go, crates.io)
  ecosystem: PyPI
```

### Tracker

The tracker `platform`, `board`, `visibility`, and `skill_url_template`
are org-level defaults — override them here only if this project differs
(e.g. a project that runs its security tracker publicly sets
`visibility: public`). Declare this project's per-tracker vocabulary:

```yaml
tracker:
  # Whether the reporter can see the tracker issue once opened.
  reporter_has_access: false

  # Whether the tracker drives a board / kanban view.
  project_board_enabled: true

  # Body-field heading names — role -> the concrete `###` heading in this
  # project's issue template. Skills refer to these by role.
  # Consumed by: every skill that reads/writes the issue body.
  body_fields:
    cve_link: "CVE tool link"
    mailing_thread: "Mailing list thread URL"
    affected_versions: "Affected versions"

  # Tracker labels — role -> the concrete label name in this project.
  # Consumed by: security-issue-triage, security-issue-sync, pr-management-triage.
  labels:
    security_marker: "security"
    needs_triage: "needs triage"
    pr_open: "pr created"
    pr_merged: "pr merged"
    cve_allocated: "cve allocated"
    not_cve_worthy: "not cve worthy"
    # Label on the single open "rejected without tracker" ledger issue
    # (see below). NOT the security_marker label.
    rejections_ledger: "rejections-ledger"
```

#### Rejected-without-tracker ledger

The `security-issue-import` skill sometimes rejects a report with a
canned reply **without creating a tracker** (the disposition lives only
on the mail thread). To keep those rejections countable, the team
records each one as a comment on a single dedicated **ledger issue** in
`tracker_repo`: one **open** issue, labelled with the
`tracker.labels.rejections_ledger` value (default `rejections-ledger`)
and **not** carrying the security-marker label.

Adopters who want the *rejected without tracker* dashboard stat must:

1. **Create the ledger issue** once in `tracker_repo` and label it
   `rejections-ledger` (keep it open; the skills resolve it via
   `gh issue list --repo <tracker> --state open --label
   rejections-ledger`).
2. **Set the dashboard knob.** Point
   `security-tracker-stats.md → rejections_ledger_label` (or the
   `rejections_ledger_label:` key in the renderer's YAML overlay) at the
   same label. Set it to `null` to disable the stat — then no ledger
   issue is needed.

Each rejection comment carries a machine-parseable block
(`<!-- rejection v1 -->` with `date:` / `reporter:` / `canned:` /
`thread:` / `summary:` lines); a one-time historical backfill is a
single `<!-- rejection-backfill v1 count: N -->` comment. The
`security-tracker-stats-dashboard` renderer parses these and excludes the
ledger issue from all tracker classification. Closes handled by
`security-issue-invalidate` are **not** ledger entries.

### Scope detection

Per-project — whether this project distinguishes scope sub-products, and
the label → sub-product map:

```yaml
scope_detection:
  # When false, every issue maps to the single product in the `product` block.
  # Consumed by: security-issue-triage, generate-cve-json, security-issue-sync.
  enabled: true

  # Scope label -> sub-product: tracker label -> CVE product / packageName
  # / upstream path-prefix the skill uses to confirm a PR touches that scope.
  labels:
    <scope-label>:
      product: "<Product Name>"
      packageName: "<package-name>"
      path_prefix: "<path-prefix-regex>"
    <secondary-scope-label>:
      product: "<Secondary Product Name>"
      packageName: "<secondary-package-name>"
      path_prefix: "<secondary-path-prefix-regex>"
```

### Release process

The release-manager lookup cascade and artifact registries are org-level
and inherited. Declare this project's stale milestones and changelog
fragment tool:

```yaml
release_process:
  # Milestones the skills treat as "stale" (overdue for re-targeting) —
  # exact milestone-name matches.
  # Consumed by: security-issue-sync, pr-management-triage.
  stale_milestones:
    - "<stale-milestone-1>"
    - "<stale-milestone-2>"

  # Whether the upstream repo uses a changelog-fragment tool, and which one.
  # Consumed by: security-issue-fix, issue-fix-workflow.
  newsfragments:
    enabled: true
    tool: <fragment-tool>
```

### Roster

The roster `source` is org-level and inherited. Declare this project's
bare-name → handle map and release managers:

```yaml
roster:
  # Mailing-list threads reference contributors by first name; this binds
  # those to GitHub handles for @-mentions.
  # Consumed by: security-issue-sync, pr-management-mentor.
  bare_name_handles:
    # Example (replace with your project's contributors):
    "<FirstName>": "@<handle>"

  # Release-manager handles, current first. Keep in sync with release-trains.md.
  # Consumed by: security-issue-sync, security-issue-fix.
  release_managers:
    # Example (replace with your project's release managers):
    - "@<handle>"
```

### Product

Per-project — the product identity used in CVE records, advisories, and
title normalization:

```yaml
product:
  # Human-readable product name — lands in the CVE record's `product` field.
  # Consumed by: generate-cve-json, canned-responses templating.
  name: <ProjectShortName>

  # Package-name shape for the primary artifact (PyPI / npm / Maven).
  # Consumed by: generate-cve-json, canned-responses templating.
  package_name: <package-name>

  # Regex matched against changed paths in an upstream PR to confirm
  # "this PR really touches the product" — a backstop in the fix flow.
  # Consumed by: security-issue-fix, pr-management-triage.
  code_pointer_path_prefix: "<code-path-prefix-regex>"

  # Prefixes the title-normalization skill strips from an inbound subject
  # when building a CVE title (matched at the start, case-insensitively).
  # Consumed by: title-normalization, generate-cve-json, canned-responses.
  subject_prefix_strip:
    - "[SECURITY]"
    - "[Security Report]"
    - "Re:"
    - "Fwd:"
    - "<vendor>:"
    - "<vendor> <product>:"

  # Prefix the affected-versions extractor strips to leave the bare version.
  # Consumed by: security-issue-sync, generate-cve-json.
  affected_version_extract_prefix: "<ProjectShortName>"
```

## Pointers to sibling files

- [`release-trains.md`](release-trains.md) — fast-moving release state, release-manager attribution, security-team roster.
- [`milestones.md`](milestones.md) — milestone naming conventions.
- [`scope-labels.md`](scope-labels.md) — scope label → CVE product mapping.
- [`security-model.md`](security-model.md) — Security-Model URL + anchors.
- [`title-normalization.md`](title-normalization.md) — CVE title strip cascade.
- [`fix-workflow.md`](fix-workflow.md) — fork / toolchain / commit-trailer specifics.
- [`naming-conventions.md`](naming-conventions.md) — project-specific editorial rules.
- [`canned-responses.md`](canned-responses.md) — reporter-facing reply templates.
- [`skill-sources.md`](skill-sources.md) — trusted external skill sources this project pulls skills from (the install gate).
- [`README.md`](README.md) — project file index + onboarding checklist.
