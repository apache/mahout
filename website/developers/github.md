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
title:

    
---

# Github Setup and Pull Requests (PRs) #

There are several ways to setup Git for committers and contributors. Contributors can safely setup 
Git any way they choose but committers should take extra care since they can push new commits to the master at 
Apache and various policies there make backing out mistakes problematic. Therefore all but very small changes should 
go through a PR, even for committers. To keep the commit history clean take note of the use of --squash below
when merging into apache/master.

## Git setup for Committers

This describes setup for one local repo and two remotes. It allows you to push the code on your machine to either your Github repo or to git-wip-us.apache.org. 
You will want to fork github's apache/mahout to your own account on github, this will enable Pull Requests of your own. 
Cloning this fork locally will set up "origin" to point to your remote fork on github as the default remote. 
So if you perform "git push origin master" it will go to github.

To attach to the apache git repo do the following:

    git remote add apache https://git-wip-us.apache.org/repos/asf/mahout.git

To check your remote setup

    git remote -v

you should see something like this:

    origin    https://github.com/your-github-id/mahout.git (fetch)
    origin    https://github.com/your-github-id/mahout.git (push)
    apache    https://git-wip-us.apache.org/repos/asf/mahout.git (fetch)
    apache    https://git-wip-us.apache.org/repos/asf/mahout.git (push)

Now if you want to experiment with a branch everything, by default, points to your github account because 'origin' is default. You can work as normal using only github until you are ready to merge with the apache remote. Some conventions will integrate with Apache Jira ticket numbers.

    git checkout -b mahout-xxxx #xxxx typically is a Jira ticket number
    #do some work on the branch
    git commit -a -m "doing some work"
    git push origin mahout-xxxx # notice pushing to **origin** not **apache**

Once you are ready to commit to the apache remote you can merge and push them directly or better yet create a PR. 

## How to create a PR (committers)

Push your branch to Github:

    git checkout mahout-xxxx
    git push origin mahout-xxxx

Go to your mahout-xxxx branch on Github. Since you forked it from Github's apache/mahout it will default
any PR to go to apache/master. 

* Click the green "Compare, review, and create pull request" button. 
* You can edit the to and from for the PR if it isn't correct. The "base fork" should be apache/mahout unless you are collaborating 
separately with one of the committers on the list. The "base" will be master. Don't submit a PR to one of the other 
branches unless you know what you are doing. The "head fork" will be your forked repo and the "compare" will be 
your mahout-xxxx branch. 
* Click the "Create pull request" button and name the request "MAHOUT-XXXX" all caps. 
This will connect the comments of the PR to the mailing list and Jira comments.
* From now on the PR lives on github's apache/mahout. You use the commenting UI there.  
* If you are looking for a review or sharing with someone else say so in the comments but don't worry about 
automated merging of your PR--you will have to do that later. The PR is tied to your branch so you can respond to 
comments, make fixes, and commit them from your local repo. They will appear on the PR page and be mirrored to Jira 
and the mailing list. 

When you are satisfied and want to push it to Apache's remote repo proceed with **Merging a PR**

## How to create a PR (contributors)

Create pull requests: \[[1]\]. 

Pull requests are made to apache/mahout repository on Github. In the Github UI you should pick the master 
branch to target the PR as described for committers. This will be reviewed and commented on so the merge is 
not automatic. This can be used for discussing a contributions in progress.

## Merging a PR (yours or contributors) 

Start with reading \[[2]\] (merging locally). 

Remember that pull requests are equivalent to a remote github branch with potentially a multitude of commits. 
In this case it is recommended to squash remote commit history to have one commit per issue, rather 
than merging in a multitude of contributor's commits. In order to do that, as well as close the PR at the 
same time, it is recommended to use **squash commits**.

Merging pull requests are equivalent to a "pull" of a contributor's branch:

    git checkout master      # switch to local master branch
    git pull apache master   # fast-forward to current remote HEAD
    git pull --squash https://github.com/cuser/mahout cbranch  # merge to master 

--squash ensures all PR history is squashed into single commit, and allows committer to use his/her own
message. Read git help for merge or pull for more information about `--squash` option. In this example we 
assume that the contributor's Github handle is "cuser" and the PR branch name is "cbranch". 
Next, resolve conflicts, if any, or ask a contributor to rebase on top of master, if PR went out of sync.

If you are ready to merge your own (committer's) PR you probably only need to merge (not pull), since you have a local copy 
that you've been working on. This is the branch that you used to create the PR.

    git checkout master      # switch to local master branch
    git pull apache master   # fast-forward to current remote HEAD
    git merge --squash mahout-xxxx

Remember to run regular patch checks, build with tests enabled, and change CHANGELOG.

If everything is fine, you now can commit the squashed request along the lines

    git commit --author <contributor_email> -a -m "MAHOUT-XXXX description closes apache/mahout#ZZ"

MAHOUT-XXXX is all caps and where `ZZ` is the pull request number on apache/mahout repository. Including 
"closes apache/mahout#ZZ" will close the PR automatically. More information is found here \[[3]\].

Next, push to git-wip-us.a.o:

    push apache master

(this will require Apache handle credentials).

The PR, once pushed, will get mirrored to github. To update your github version push there too:

    push origin master

*Note on squashing: Since squash discards remote branch history, repeated PRs from the same remote branch are 
difficult for merging. The workflow implies that every new PR starts with a new rebased branch. This is more 
important for contributors to know, rather than for committers, because if new PR is not mergeable, github
would warn to begin with. Anyway, watch for dupe PRs (based on same source branches). This is a bad practice.*
     
## Closing a PR without committing (for committers)

When we want to reject a PR (close without committing), we can just issue an empty commit on master's HEAD 
*without merging the PR*: 

    git commit --allow-empty -m "closes apache/mahout#ZZ *Won't fix*"
    git push apache master

that should close PR `ZZ` on github mirror without merging and any code modifications in the master repository.

## Apache/github integration features 

Read \[[4]\]. Comments and PRs with Mahout issue handles should post to mailing lists and Jira.
Mahout issue handles must be in the form MAHOUT-YYYYY (all capitals). Usually it makes sense to 
file a jira issue first, and then create a PR with description 
    
    MAHOUT-YYYY: <jira-issue-description>


In this case all subsequent comments will automatically be copied to jira without having to mention 
jira issue explicitly in each comment of the PR.


[1]: https://help.github.com/articles/creating-a-pull-request
[2]: https://help.github.com/articles/merging-a-pull-request#merging-locally
[3]: https://help.github.com/articles/closing-issues-via-commit-messages
[4]: https://blogs.apache.org/infra/entry/improved_integration_between_apache_and
