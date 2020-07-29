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
# Handling Github PRs #

----------

Hit the merge button.

## how to create a PR (for contributers)

Read [[1]]. 

Pull requests are made to apache/mahout repository on Github. 

## merging a PR and closing it (for committers). 

Remember that pull requests are equivalent to a remote branch with potentially a multitude of commits. 
In this case it is recommended to squash remote commit history to have one commit per issue, rather 
than merging in a multitude of contributer's commits. In order to do that, as well as close the PR at the 
same time, it is recommended to use **squash commits**.

Read [[2]] (merging locally). Merging pull requests are equivalent to merging contributor's branch:

    git checkout master      # switch to local master branch
    git pull apache master   # fast-forward to current remote HEAD
    git pull --squash https://github.com/cuser/mahout cbranch  # merge to master 


In this example we assume that contributor Github handle is "cuser" and the PR branch name is "cbranch" there. We also 
assume that *apache* remote is configured as 

    apache  https://git-wip-us.apache.org/repos/asf/mahout.git (fetch)
    apache  https://git-wip-us.apache.org/repos/asf/mahout.git (push)


Squash pull ensures all PR history is squashed into single commit. Also, it is not yet committed, even if 
fast forward is possible, so you get chance to change things before committing.

At this point resolve conflicts, if any, or ask contributor to rebase on top of master, if PR went out of sync.

Also run regular patch checks and change CHANGELOG.

Suppose everything is fine, you now can commit the squashed request 

    git commit -a

edit message to contain "MAHOUT-YYYY description **closes #ZZ**", where ZZ is the pull request number. 
Including "closes #ZZ" will close PR automatically. More information [[3]].

   push apache master

(this will require credentials).

Note on squashing: Since squash discards remote branch history, repeated PRs from the same remote branch are 
difficult for merging. The workflow implies that every new PR starts with a new rebased branch. This is more 
important for contributors to know, rather than for committers, because if new PR is not mergeable, github
would warn to begin with. Anyway, watch for dupe PRs (based on same source branches). This is a bad practice.
     
## Closing a PR without committing 

When we want to reject a PR (close without committing), just do the following commit on master's HEAD 
*without merging the PR*: 

    git commit --allow-empty -m "closes #ZZ *Won't fix*"
    git push apache master

that should close PR without merging and any code modifications in the master repository.

## Apache/github integration features 

Read [[4]]. Issue handles mentioned in comments and PR name should post to mailing lists and Jira.


[1]: https://help.github.com/articles/creating-a-pull-request
[2]: https://help.github.com/articles/merging-a-pull-request#merging-locally
[3]: https://help.github.com/articles/closing-issues-via-commit-messages
[4]: https://blogs.apache.org/infra/entry/improved_integration_between_apache_and
