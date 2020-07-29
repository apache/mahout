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
title: Issue Tracker

    
---

# Issue tracker


Mahout's issue tracker is located [here](http://issues.apache.org/jira/browse/MAHOUT).
For most changes (apart from trivial stuff) Mahout works according to a review-then-commit model.
This means anything that is to be added is first presented as a patch in the issue tracker. All conversations in the issue tracker are automatically
echoed on the developer mailing list and people tend to respond or continue
conversations there rather in the issue tracker, so in order to follow an
issue you also have to read to the <a href="http://mahout.apache.org/general/mailing-lists,-irc-and-archives.html">mailing lists</a>. 

An issue does not literally have to be an issue. It could be a wish, task,
bug report, etc. and it does not have to contain a patch.

Mahout uses [JIRA](https://confluence.atlassian.com/display/JIRA/JIRA+Documentation) by Atlassian.

<a name="IssueTracker-Bestpractise"></a>
#### Best practices

Don't create duplicate issues. Make sure your problem is a problem and that
nobody else already fixed it. If you are new to the project, it is often
preferred that the subject of an issue is discussed on one of our mailing
lists before an issue is created - in particular when it comes to adding new functionality.

Quote only what it is you are responding to in comments.

Patches should be created at trunk or trunk parent level and if possible be
a single uncompressed text file so it is easy to inspect the patch in a web
browser. (See [Patch Check List](/developers/patch-check-list.html)
.)

Use the issue identity when referring to an issue in any discussion.
"MAHOUT-n" and not "mahout-n" or "n". MAHOUT-1 would automatically be
linked to [MAHOUT-1](http://issues.apache.org/jira/browse/MAHOUT-1)
 in a better world.

A note to committers: Make sure to mention the issue id in each commit. Not only has
JIRA the capability of auto-linking commits to the issue they are related to
that way, it also makes it easier to get further information for a specific commit
when browsing through the commit log and within the commit mailing list.
