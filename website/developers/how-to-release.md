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
layout: page
title: How To Release

    
---

# How To Release Mahout


*This page is prepared for Mahout committers. You need committer rights to
create a new Mahout release.*

<a name="HowToRelease-ReleasePlanning"></a>
# Release Planning

Start a discussion on mahout-dev about having a release, questions to bring
up include:

 * Any [Unresolved JIRA issues for the upcoming release ](-https://issues.apache.org/jira/secure/issuenavigator!executeadvanced.jspa?jqlquery=project+%3d+mahout+and+resolution+%3d+unresolved+and+fixversion+%3d+%220.6%22&runquery=true&clear=true.html)
 * Any [Resolved or Closed JIRA issues missing a "Fix Version" ](-https://issues.apache.org/jira/secure/issuenavigator!executeadvanced.jspa?jqlquery=project+%3d+mahout+and+%28status+%3d+resolved+or+status+%3d+closed%29+and+fixversion+is+null+and+resolution+%3d+fixed&runquery=true&clear=true.html)
 that should be marked as fixed in this release?
 * Does any documentation need an update?
 * Who is going to be the release engineer?
 * What day should be targeted for the release? Leave buffer time for a code freeze and release candidate testing; make sure at least a few people commit to having time to help test the release candidates around the target date.


<a name="HowToRelease-CodeFreeze"></a>
# Code Freeze

For 7-14 days prior to the release target date, have a "code freeze" where
committers agree to only commit things if they:

 * Are documentation improvements (including fixes to eliminate Javadoc
warnings)
 * Are new test cases that improve test coverage
 * Are bug fixes found because of improved test coverage
 * Are new tests and bug fixes for new bugs encountered by manually testing

<a name="HowToRelease-StepsForReleaseEngineer"></a>
# Steps For Release Engineer

<a name="HowToRelease-Beforebuildingrelease"></a>
## Before building release
1. Check that all tests pass after a clean compile: `mvn clean test`
1. Check that there are no remaining unresolved Jira issues with the upcoming version number listed as the "Fix" version
1. Publish any previously unpublished third-party dependenciess: [Thirdparty Dependencies](thirdparty-dependencies.html)

<a name="HowToRelease-Makingarelease"></a>
## Making a release
* If this is your first release, add your key to the KEYS file (located on GitHub at https://github.com/apache/mahout/master/distribution/KEYS) and copy it to the release directory.  Make sure you commit your change.  See http://www.apache.org/dev/release-signing.html.
* Ensure you have set up standard Apache committer settings in
 ~/.m2/settings.xml as per [this page](http://maven.apache.org/developers/committer-settings.html)
.
* Add a profile to your `~/.m2/settings.xml` in the `<profiles>` section with:

```
<profiles>
  <profile>
    <id>mahout_release</id>
    <properties>
      <gpg.keyname>1234BEEF</gpg.keyname>
      <gpg.passphrase>YOUR_SIGNING_PASSCODE_HERE</gpg.passphrase>
      <deploy.altRepository>mahout.releases::default::https://repository.apache.org/service/local/staging/deploy/maven2/</deploy.altRepository>
      <username>USERNAME</username>
      <deploy.url>https://repository.apache.org/service/local/staging/deploy/maven2/</deploy.url>
    </properties>
  </profile>
</profiles>
```

* You may also need to add the following to the `<servers>` section in `~/.m2/settings.xml` in order to upload artifacts (as the `-Dusername=` `-Dpassword=` didn't work for gsingers for 0.8, but this did; n.b. it didn't work for akm for the 0.13 release):

```
<server>
  <id>apache.releases.https</id>
  <username>USERNAME</username>
  <password>PASSWORD</password>
</server>
```


* If you are outside the US, then svn.apache.org may not resolve to the main US-based Subversion servers. (Compare the IP address you get for svn.apache.org with svn.us.apache.org to see if they are different.) This will cause problems during the release since it will create a revision and then immediately access, but, there is a replication lag of perhaps a minute to the non-US servers. To temporarily force using the US-based server, edit your equivalent of /etc/hosts and map the IP address of svn.us.apache.org to svn.apache.org.
* Create the release candidate: `mvn -Pmahout-release,apache-release release:prepare release:perform`; to add credentials for source control use `mvn -Dusername=myusername -Dpassword='mypassword' -Papache-release release:prepare release:perform`

* If you have problems authenticating to svn.apache.org, try adding to the command line `-Dusername=USERNAME -Dpassword=PASSWORD`
* If there is an issue first try doing: 
  1. `mvn -Dmahout-release,apache-release release:rollback` 
  1. followed by `mvn -Dmahout-release,apache-release release:clean` as this will likely save you time and do the right thing. You may also have to rollback the version numbers in the POM files. 
  1. _Optional_ If you want to skip test cases while rebuilding, use `mvn -DpreparationGoals="clean compile" release:prepare release:perform`
* Review the artifacts, etc. on the Apache Repository (using Sonatype's Nexus application) site: https://repository.apache.org/. You will need to login using your ASF SVN credentials and then browse to the staging area.
* Once you have reviewed the artifacts, you will need to "Close" out
the staging area under Nexus, which then makes the artifacts available for
others to see.
  1. Log in to Nexus
  1. Click the Staging Repositories link in the left hand menu
  1. Click the Mahout staged artifact that was just uploaded by the
release:perform target
  1. Click Close in the toolbar. See https://docs.sonatype.org/display/Repository/Closing+a+Staging+Repository for a picture
  1. Copy the "Repository URL" link to your email; it should be formed like https://repository.apache.org/content/repositories/orgapachemahout-024/
* Call a VOTE on dev@mahout.apache.org.  Votes require 3 days before passing.  See Apache [release policy|http://www.apache.org/foundation/voting.html#ReleaseVotes] for more info.
* If there's a problem, you need to unwind the release and start all over.

```
mvn -Pmahout-release,apache-release versions:set -DnewVersion=PREVIOUS_SNAPSHOT
mvn -Pmahout-release,apache-release versions:commit
git commit 
git push --delete apache <tagname> (deletes the remote tag)
git tag -d tagname (deletes the local tag)
```

* Release the artifact in the Nexus Repository in the same way you closed it earlier.
* Per http://www.apache.org/dev/release-publishing.html#distribution_dist move artifacts over to dist.apache.org
* Get the Subversion (SVN) repo for Mahout at https://dist.apache.org/repos/dist/release/mahout
* Make sure to add your key to KEYS if it's not there already: https://dist.apache.org/repos/dist/release/mahout/KEYS
* Copy the assemblies and their supporting files (tar.gz, zip, tar.bz2, plus .asc, .md5, .pom, .sha1 files) to a new directory and check in the new directory and the edited KEYS file if you changed it
* Wait 24 hours for release to propagate to mirrors.
* Clean up JIRA: Bulk close all X.Y JIRA issues.  Mark the Version
number as being released (see Manage Versions.)  Add the next version
(X.Y+1) if necessary.
* Update release version on http://mahout.apache.org/ and
http://en.wikipedia.org/wiki/Apache_Mahout
* Send announcements to the user and developer lists.


See also:

* http://maven.apache.org/developers/release/releasing.html
*
http://www.sonatype.com/books/nexus-book/reference/staging-sect-deployment.html
* http://www.sonatype.com/books/nexus-book/reference/index.html


### Post Release
## Versioning
* Create the next version in JIRA (if it doesn't already exist)
* Mark the version as "released" in JIRA (noting the release date)

## Documentation
* Change wiki to match current best practices (remove/change deprecations,
etc)

## Publicity
* update freshmeat
* blog away
* Update MLOSS entry: http://mloss.org/revision/view/387/.  See Grant for
details.

## Related Resources

* http://www.apache.org/dev/#releases
* http://www.apache.org/dev/#mirror

# TODO: Things To Cleanup in this document

* more specifics about things to test before starting or after packaging
(RAT, run scripts against example, etc...)
* include info about [Voting | http://www.apache.org/foundation/voting.html#ReleaseVotes]
