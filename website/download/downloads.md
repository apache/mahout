---
layout: page
title: Downloads


---

<a name="Downloads-OfficialRelease"></a>
# Official Release
Apache Mahout is an official Apache project and thus available from any of
the Apache mirrors. The latest Mahout release is available for download at: 

* [Download Latest](http://www.apache.org/dist/mahout)
* [Release Archive](http://archive.apache.org/dist/mahout)
  * To validate artifacts:
    * (From KEYS file): `gpg --import KEYS`
```
$ gpg mahout-qumat-0.4.zip.gpg
gpg: assuming signed data in `mahout-qumat-0.4.zip.gpg'
gpg: Signature made Fri 01 Mar 2019 09:59:00 AM PST using RSA key ID 140A5BE9
gpg: Good signature from "Apache B. Committer (ASF Signing Key) <abc@apache.org>"
```

# Source code for the current snapshot

Apache Mahout is mirrored to [Github](https://github.com/apache/mahout). To get all source:

    git clone https://github.com/apache/mahout.git mahout

## Getting started

To install dependencies, run the following:
```
pip install -U poetry
poetry install
```

<a name="Downloads-FutureReleases"></a>
# Future Releases

Official releases are usually created when the developers feel there are
sufficient changes, improvements and bug fixes to warrant a release. Watch
the <a href="https://mahout.apache.org/community/mailing-lists.html">Mailing lists</a>
 for latest release discussions and check the Github repo.

