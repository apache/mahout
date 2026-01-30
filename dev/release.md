# Release Process for Qumat and Qumat-QDP (via ATR)

This document describes the process for releasing `qumat` and `qumat-qdp`. The process is divided into three main phases:
1.  **Community Pre-Release**: Preparation and testing of the Release Candidate (RC) by the community.
2.  **Official Release**: Formal signing, voting, and distribution of source artifacts using the **Apache Trusted Releases (ATR)** platform.
3.  **Final Publication**: Publishing the final version to PyPI after the vote passes.

## Prerequisites

-   **ASF Account**: Required for PMC members to log in to the ATR platform.
-   **PyPI Account**: Required for uploading RCs for community testing.
-   **GPG Key**: Required for signing release artifacts. See [GPG Key Setup](#gpg-key-setup) below.

### GPG Key Setup

If you don't have a GPG key yet, generate one:

```bash
# Generate a 4096-bit RSA key (use your Apache email)
gpg --full-generate-key
# Select: RSA and RSA, 4096 bits, your_name <your_id@apache.org>

# Find your key ID
gpg --list-secret-keys --keyid-format SHORT
# Example output: rsa4096/8EEFC01F

# Upload your public key to a key server
gpg --keyserver keys.openpgp.org --send-keys <YOUR_KEY_ID>

# Append your public key to the project KEYS file
(gpg --list-sigs <YOUR_KEY_ID> && gpg --armor --export <YOUR_KEY_ID>) >> KEYS
```

### PyPI Token Setup

**Important:** Store your `.pypirc` in your **home directory** (`~/.pypirc`), **never** in the project directory — it can accidentally be included in source distributions.

```bash
# Create ~/.pypirc (not in the project directory!)
cat > ~/.pypirc << 'EOF'
[testpypi]
username = __token__
password = pypi-xxxxx

[pypi]
username = __token__
password = pypi-xxxxx
EOF

chmod 600 ~/.pypirc
```

---

## Branching Strategy

We follow the Airflow-style release branching model:

```
main ────●────●────●────●────●──→ (development continues)
                   │
                   ├── mahout-qumat-0.5.0-RC1 (tag)
                   │
                   └── v0.5-stable (branch) ──●──→ RC2 ──→ v0.5.0 ──→ v0.5.1
```

### Create Stable Branch

When ready to cut a release, create a stable branch from `main`:

```bash
git checkout main
git pull upstream main
git checkout -b v0.5-stable
git push -u upstream v0.5-stable
```

### Tag Release Candidates

Tag RCs on the stable branch:

```bash
git checkout v0.5-stable
git tag -a mahout-qumat-0.5.0-RC1 -m "Release Candidate 1 for qumat 0.5.0"
git push upstream mahout-qumat-0.5.0-RC1
```

### Cherry-pick Bug Fixes

If bugs are found during RC testing:

1. **Fix on `main` first** (keeps main up-to-date):
   ```bash
   git checkout main
   # ... fix and commit ...
   git push upstream main
   ```

2. **Cherry-pick to stable branch**:
   ```bash
   # Using cherry-picker tool (auto-creates PR)
   uvx cherry-picker <commit-hash> v0.5-stable
   ```

3. **Tag new RC**:
   ```bash
   git tag -a mahout-qumat-0.5.0-RC2 -m "Release Candidate 2 for qumat 0.5.0"
   git push upstream mahout-qumat-0.5.0-RC2
   ```

---

## Phase 1: Community Pre-Release (RC Preparation)

The goal of this phase is to ensure the release candidate is stable and ready for a formal vote.

### 1.1 Plan the Release
-   Discuss and decide on the release timeline (e.g., "RC1 target date") on the `dev@mahout.apache.org` mailing list or Slack.

### 1.2 Prepare Artifacts
Update version numbers and build the artifacts locally.

**Update Versions to RC:**
Ensure the version includes the `rc` suffix (e.g., `0.5.0rc1`):
-   `pyproject.toml` — set `version = "0.5.0rc1"`
-   `qdp/Cargo.toml` — set `version = "0.1.0-rc1"`

**Build:**
```bash
# Build Qumat (pure Python — one wheel for all Python versions)
uv build

# Build Qumat-QDP (native Rust — one wheel per Python version)
cd qdp/qdp-python
uv tool run maturin build --release --interpreter python3.10
uv tool run maturin build --release --interpreter python3.11
uv tool run maturin build --release --interpreter python3.12
```

**Output locations:**
-   `dist/qumat-0.5.0rc1-py3-none-any.whl`
-   `dist/qumat-0.5.0rc1.tar.gz`
-   `qdp/target/wheels/qumat_qdp-0.1.0rc1-cp3XX-*.whl`

### 1.3 Sign and Hash Artifacts

Sign each artifact with your GPG key and generate SHA-512 checksums:

```bash
# Sign and hash Qumat artifacts
cd dist
for f in qumat-0.5.0rc1*; do
  gpg --armor --detach-sign "$f"
  sha512sum "$f" > "$f.sha512"
done

# Sign and hash Qumat-QDP wheels
cd ../qdp/target/wheels
for f in qumat_qdp-0.1.0rc1-*.whl; do
  gpg --armor --detach-sign "$f"
  sha512sum "$f" > "$f.sha512"
done
```

This produces `.asc` (GPG signature) and `.sha512` (checksum) files for each artifact. These are required by ATR for release validation.

**Verify signatures locally:**
```bash
gpg --verify qumat-0.5.0rc1.tar.gz.asc qumat-0.5.0rc1.tar.gz
```

### 1.4 Upload to PyPI (RC Version)

Ensure `~/.pypirc` is configured (see [PyPI Token Setup](#pypi-token-setup)).

**Upload to TestPyPI first (recommended):**
```bash
# Upload Qumat
uv tool run twine upload --repository testpypi --config-file ~/.pypirc dist/*

# Upload Qumat-QDP
uv tool run twine upload --repository testpypi --config-file ~/.pypirc qdp/target/wheels/qumat_qdp-<version>-cp31{0,1,2}-*.whl
```

**Test install from TestPyPI:**
```bash
uv venv && source .venv/bin/activate
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  qumat==0.5.0rc1 qumat-qdp==0.1.0rc1
pytest testing/
```

**Upload to PyPI:**
```bash
# Upload Qumat
uv tool run twine upload --repository pypi --config-file ~/.pypirc dist/*

# Upload Qumat-QDP (exclude Python versions outside requires-python)
uv tool run twine upload --repository pypi --config-file ~/.pypirc qdp/target/wheels/qumat_qdp-<version>-cp31{0,1,2}-*.whl
```

*Note: This makes the RC available on PyPI for testing with `pip install --pre qumat==0.5.0rc1`.*

### 1.5 Open Testing Issue
Use the script to generate the RC testing issue:

```bash
# Generate issue content (dry run)
./dev/generate-rc-issue.sh 0.5.0rc1 0.1.0rc1 "Qumat 0.5.0"

# Create GitHub issue directly
./dev/generate-rc-issue.sh 0.5.0rc1 0.1.0rc1 "Qumat 0.5.0" | \
  gh issue create --repo apache/mahout \
    --title "Status of testing Apache Mahout Qumat 0.5.0rc1" \
    --body-file -
```

The script generates an issue with:
-   Installation commands
-   All PRs from the milestone with checkboxes
-   Contributor mentions for testing

### 1.6 Community Testing & Closure
-   Allow a testing interval (e.g., 3-5 days).
-   If critical bugs are found: Fix them, increment the RC number (e.g., RC2), and repeat from Step 1.2.
-   Once the community is satisfied and the issue shows positive feedback, close the issue and proceed to Phase 2.

---

## Phase 2: PMC Official Release via ATR

This phase is executed by a PMC member or Release Manager using the ATR platform. The PMC votes on **source artifacts**, which are the canonical release artifacts for Apache projects.

### 2.1 Submit Source Artifacts to ATR
1.  Navigate to the [ATR Web UI](https://release-test.apache.org/).
2.  Select the **Mahout** project.
3.  Click **"Start New Release"**.
4.  Upload the **source tarballs** from Phase 1.2:
    -   `qumat/dist/qumat-1.0.0.tar.gz`
    -   `qdp/qdp-python/target/wheels/qumat_qdp-1.0.0.tar.gz`
    -   You must sign artifacts and generate checksums yourself before uploading (see [Step 1.3](#13-sign-and-hash-artifacts)).

### 2.2 Verify Release
Check the "Release Candidates" section in ATR to ensure:
-   Source artifacts are correct
-   GPG signatures are valid
-   Checksums are generated

### 2.3 Vote
Use the ATR platform to generate the Vote email template.
1.  In the ATR UI, find the Release Candidate.
2.  Click **"Generate Vote Email"** or **"Start Vote"**.
3.  Send the email to `dev@mahout.apache.org`.
4.  Wait for the standard 72-hour voting period (requiring 3 binding +1 votes from Committers or PMC Members).

---

## Phase 3: Final Publication

### 3.1 Finalize Release in ATR
Once the vote passes:
1.  Go to the Release Candidate in the ATR UI.
2.  Click **"Promote to Release"** (or **"Publish"**).
    -   This moves the signed source artifacts to the Apache release SVN at `https://dist.apache.org/repos/dist/release/mahout/`.

### 3.2 Publish Final Version to PyPI
After the PMC vote passes, publish the final version to PyPI. The artifacts must be built from the voted source in Apache SVN to ensure they match what the PMC approved.

**Automated Process:**
Set up a GitHub Actions workflow that:
-   Is triggered manually via "Workflow Dispatch" with the release version as input.
-   Downloads the approved source tarball from Apache SVN (`https://dist.apache.org/repos/dist/release/mahout/`).
-   Verifies GPG signatures and checksums.
-   Builds wheels from the verified source.
-   Uploads to PyPI using Trusted Publisher configuration.

**Important:** The PyPI release is built from the exact source code stored in Apache SVN that the PMC voted on, ensuring consistency between the Apache release and PyPI.

### 3.3 Post-Release Actions
-   **Tag the release** in Git:
    ```bash
    git tag -a v1.0.0 -m "Release 1.0.0"
    git push origin v1.0.0
    ```
-   **Bump the version** in `main` to the next development version (e.g., `1.1.0.dev0`).
-   **Create versioned documentation**:
    ```bash
    cd website
    npm run version 1.0.0
    ```
    This creates a snapshot of the current docs under `versioned_docs/version-1.0.0/` and adds the version to `versions.json`. Update `docusaurus.config.ts` to configure the new version label and path if needed.
-   **Announce the release** on `dev@mahout.apache.org`.
-   **Update website documentation** with the new release notes.
