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
title: How To Update The Website
---

# How to update the Mahout Website

Committers and contributors are all welcomed and encouraged to update the Mahout website.
 Mahout uses Jekyll to build the website. A [script](https://github.com/apache/mahout/blob/master/website/build_site.sh)
 is executed by Jenkins after any change is detected in the `master` branch of the official Apache repository (
 e.g. after any PR is merged ). As such it is important that you, and any reviewers stage site changes locally before
 committing. 
 
 The process for updating the website is as follows:
 
#### Clone the Mahout Git Repository Locally

```git clone http://github.com/apache/mahout```

See [Building from Source](/developers/buildingmahout/#getting-code) for more details.

#### Working with Markdown

Jekyll uses [Kramdown](https://kramdown.gettalong.org/syntax.html) to compile markdown into HTML. 

Kramdown syntax is very similar to standard markdown, but with a few subtle diferences, please review
to the [Kramdown syntax guide](https://kramdown.gettalong.org/syntax.html).

Mahout is a highly mathematical project, and contributors will often want to use [LaTeX Mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
to explain concepts.  While in some environments this is signalled to the compiler with `\( ... \)` in Kramdown
LaTeX portions are signalled with enclosing `$` characters, e.g. `$$ ... $$`

#### Stage changes locally

This step actually can be done _while you are editing_. Then you can see you changes in near-real time 
(pending browser refreshes). 

In the terminal,

```bash
cd wesite
bundle exec jekyll serve
```

By default this will serve the website locally at [http://127.0.0.1:4000/](http://127.0.0.1:4000/). You can
open your favorite browser and make sure your changes look the way you expect them to.

#### Commit code and open a PR

Once you're sure everything is right, you commit your code, push to your github.com account (preferably on a branch other than `master`
then click "OpenPR"). This process closely follows [How To Contribute- Making Changes](/developers/how-to-contribute/#HowToContribute-MakingChanges) with an exception that for _WEBSITE ONLY_ changes we relax the requirement to open a JIRA ticket. That is to say, small
website changes such as fixing a broken link or typo, do not require a specific JIRA issues, and where you would normally 
commit with a message like `MAHOUT-XXXX The thing I did` (where `XXXX` is the assosciated JIRA number), you can instead 
simply create a message like `WEBSITE Typos in how-to-update-the-website.md`.  There's nothing to stop you from making a 
JIRA issue, it simply isn't required. 

The same goes for when you open a PR (where conventionally one includes the JIRA issue, you can again title `WEBSITE` to indicate
there is no JIRA)

#### Wait for review

A committer will be along shortly to review your changes, please be patient. In the meantime, feel free to help us out by reviewing
other contributors PRs. (Here's a little secret, this is a great way to signal to us that you're interested in becoming a committer too,
as PR reviews is a big part of a committer's job).

Once everything is confirmed to be in order, the committer will merge your pull request. 

#### Committers ONLY

No further action is needed, this section is here to deliniate from the old CMS system and Jekyll builds of other projects. Jenkins
will execute [build_site.sh](https://github.com/apache/mahout/blob/master/website/build_site.sh) upon merging. This will build the website and
copy it to the `asf-site` branch, where [mahout.apache.org](http://mahout.apache.org) is served from. 