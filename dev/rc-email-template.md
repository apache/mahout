# RC Announcement Email Template

Send this to `dev@mahout.apache.org` when cutting a release candidate.

---

**Subject:** [ANNOUNCE] Qumat {VERSION} RC{N} - Cutting Release Candidate

---

Hi all,

The main features for Qumat {VERSION} have reached a good stopping point, and
I'm ready to start the release process. As the Release Manager for this
release, I'm planning to cut RC{N} on {DATE} at {TIME} (UTC+8).

Milestone: https://github.com/apache/mahout/milestone/{MILESTONE_NUMBER}

Please let me know if there are any blockers or concerns before I proceed.

Thanks,
{YOUR_NAME}

---

# RC Voting Email Template

Send this to `dev@mahout.apache.org` after the RC has been published to PyPI.

---

**Subject:** [VOTE] Release Mahout qumat from {QUMAT_VERSION}rc{N} and qumat-qdp from {QDP_VERSION}rc{N}

---

Hi all,

I have created a release candidate for Mahout qumat {QUMAT_VERSION}rc{N} and qumat-qdp {QDP_VERSION}rc{N}.

PyPI
- qumat: https://pypi.org/project/qumat/{QUMAT_VERSION}rc{N}/
- qumat-qdp: https://pypi.org/project/qumat-qdp/{QDP_VERSION}rc{N}/
GitHub Tag https://github.com/apache/mahout/releases/tag/mahout-qumat-{QUMAT_VERSION}-RC{N}
Changelog https://github.com/apache/mahout/issues/{ISSUE_NUMBER}

To test the release candidate:

```bash
pip install --pre "qumat=={QUMAT_VERSION}rc{N}"
pip install --pre "qumat-qdp=={QDP_VERSION}rc{N}"
```

Please vote on the release. The vote will be open for at least 72 hours.

[ ] +1 Release these packages as Apache Mahout qumat {QUMAT_VERSION} and qumat-qdp {QDP_VERSION}
[ ] +0 No opinion
[ ] -1 Do not release (please provide reason)

Thanks,
{YOUR_NAME}
