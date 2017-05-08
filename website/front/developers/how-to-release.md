---
layout: default
title: How to Release Mahout
theme: 
    name: mahout2
---


Title: How To Release

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
 * Who is going to be the "release engineer"?
 * What day should be targeted for the release ?  Leave buffer time for a
code freeze and release candidate testing; make sure at least a few people
commit to having time to help test the release candidates around the target
date.


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
1. Check that all tests pass after a clean compile: mvn clean test
1. Check that there are no remaining unresolved Jira issues with the
upcoming version number listed as the "Fix" version
1. Publish any prev. unpublished Third Party Deps: [Thirdparty Dependencies](thirdparty-dependencies.html)

<a name="HowToRelease-PreviewingtheArtifacts"></a>
## Previewing the Artifacts
1. To build the artifacts:
1. # mvn -Pmahout-release,apache-release,hadoop2 package

<a name="HowToRelease-Makingarelease"></a>
## Making a release
* Check if documentation needs an update
* Update the web site's news by updating a working copy of the SVN
directory at https://svn.apache.org/repos/asf/mahout/site/new_website
* Commit these changes. It is important to do this prior to the build so
that it is reflected in the copy of the website included with the release
for documentation purposes.
* If this is your first release, add your key to the KEYS file. The KEYS
file is located on Github at
https://github.com/apache/mahout/master/distribution/KEYS and copy it
to the release directory. 
Make sure you commit your change.
See http://www.apache.org/dev/release-signing.html.
* Ensure you have set up standard Apache committer settings in
 ~/.m2/settings.xml as per [this page](http://maven.apache.org/developers/committer-settings.html)
.
* Add a profile to your ~/.m2/settings.xml in the <profiles> section with:

 <blockquote>
  <profiles>
    <profile>
      <id>mahout_release</id>
      <properties>
	<gpg.keyname>YOUR PGP KEY NAME</gpg.keyname>
	<gpg.passphrase>YOUR SIGNING PASSCODE HERE</gpg.passphrase>
       
<deploy.altRepository>mahout.releases::default::https://repository.apache.org/service/local/staging/deploy/maven2/</deploy.altRepository>
	<username>USERNAME</username>
       
<deploy.url>https://repository.apache.org/service/local/staging/deploy/maven2/</deploy.url>
      </properties>
    </profile>
  </profiles>
</blockquote>

* You may also need to add the following to the <servers> section in
~/.m2/settings.xml in order to upload artifacts (as the -Dusername=
-Dpassword= didn't work for Grant for 0.8, but this did):
<blockquote>
<server>
  <id>apache.releases.https</id>
  <username>USERNAME</username>
  <password>PASSWORD</password>
</server>
</blockquote>

* Set environment variable MAVEN_OPTS to -Xmx1024m to ensure the tests can
run
* export _JAVA_OPTIONS="-Xmx1g"
* If you are outside the US, then svn.apache.org may not resolve to the
main US-based Subversion servers. (Compare the IP address you get for
svn.apache.org with svn.us.apache.org to see if they are different.) This
will cause problems during the release since it will create a revision and
then immediately access, but, there is a replication lag of perhaps a
minute to the non-US servers. To temporarily force using the US-based
server, edit your equivalent of /etc/hosts and map the IP address of
svn.us.apache.org to svn.apache.org.
* Create the release candidate:

     mvn -Pmahout-release,apache-release,hadoop2 release:prepare release:perform 

  If you have problems authenticating to svn.apache.org, try adding to the command line 

      -Dusername=\[user]\ -Dpassword=\[password\]

 If it screws up, first try doing:

  mvn -Dmahout-release,apache-release,hadoop2 release:rollback. 

 followed by

  mvn -Dmahout-release,apache-release,hadoop2 release:clean

 This will likely save you time and do the right thing. You may also need to delete the tag in source control:

  git tag -d mahout-X.XX.X; git push apache :refs/tags/mahout-X.XX.X

 You may also have to rollback the version numbers in the POM files.

 If you want to skip test cases while rebuilding, use

 mvn -DpreparationGoals="clean compile" release:prepare release:perform

* Review the artifacts, etc. on the Apache Repository (using Sonatype's
Nexus application) site: https://repository.apache.org/.
           You will need to login using your ASF SVN credentials and then
browse to the staging area.
* Once you have reviewed the artifacts, you will need to "Close" out
the staging area under Nexus, which then makes the artifacts available for
others to see.
     * Log in to Nexus
     * Click the Staging Repositories link in the left hand menu
     * Click the Mahout staged one that was just uploaded by the
release:perform target
     * Click Close in the toolbar. See
https://docs.sonatype.org/display/Repository/Closing+a+Staging+Repository
for a picture
     * Copy the "Repository URL" link to your email; it should be like
https://repository.apache.org/content/repositories/orgapachemahout-024/
* Call a VOTE on dev@mahout.apache.org.  Votes require 3 days before
passing.  See Apache [release policy|http://www.apache.org/foundation/voting.html#ReleaseVotes]
 for more info.
* If there's a problem, you need to unwind the release and start all
over.
        <blockquote>
        mvn -Pmahout-release,apache-release,hadoop2 versions:set -DnewVersion=PREVIOUS_SNAPSHOT

        mvn -Pmahout-release,apache-release,hadoop2 versions:commit

        git commit 

        git push --delete apache <tagname> (deletes the remote tag)
        git tag -d tagname (deletes the local tag)

* Release the artifact in the Nexus Repository in the same way you
Closed it earlier.
* Add your key to the KEYS file at
http://www.apache.org/dist/mahout/<version>/
* Copy the assemblies and their supporting files (tar.gz, zip, tar.bz2,
plus .asc, .md5, .pom, .sha1 files) to the ASF mirrors at:
people.apache.org:/www/www.apache.org/dist/mahout/<version>/. You should
make sure the group "mahout" owns the files and that they are read only
(-r--r--r-- in UNIX-speak). See [Guide To Distributing Existing Releases Through The ASF Mirrors|http://jakarta.apache.org/site/convert-to-mirror.html?Step-By-Step]
 and the links that are there.
     * cd /www/www.apache.org/dist/mahout
     * mkdir <VERSION>
     * cd <VERSION>
     * wget -e robots=off --no-check-certificate -np -r
https://repository.apache.org/content/groups/public/org/apache/mahout/apache-mahout-distribution/<VERSION>/
     * mv
repository.apache.org/content/groups/public/org/apache/mahout/mahout-distribution/0.8/*
.
     * rm -rf repository.apache.org/
     * rm index.html
* Wait 24 hours for release to propagate to mirrors.
* Clean up JIRA: Bulk close all X.Y JIRA issues.  Mark the Version
number as being released (see Manage Versions.)  Add the next version
(X.Y+1) if necessary.
* Update release version on http://mahout.apache.org/ and
http://en.wikipedia.org/wiki/Apache_Mahout
*
https://cwiki.apache.org/confluence/display/MAHOUT/How+To+Update+The+Website
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