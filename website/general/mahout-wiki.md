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
layout: default
title: Mahout Wiki

    
---
Apache Mahout is a new Apache TLP project to create scalable, machine
learning algorithms under the Apache license. 

{toc:style=disc|minlevel=2}

<a name="MahoutWiki-General"></a>
## General
[Overview](overview.html)
 -- Mahout? What's that supposed to be?

[Quickstart](quickstart.html)
 -- learn how to quickly setup Apache Mahout for your project.

[FAQ](faq.html)
 -- Frequent questions encountered on the mailing lists.

[Developer Resources](developer-resources.html)
 -- overview of the Mahout development infrastructure.

[How To Contribute](how-to-contribute.html)
 -- get involved with the Mahout community.

[How To Become A Committer](how-to-become-a-committer.html)
 -- become a member of the Mahout development community.

[Hadoop](http://hadoop.apache.org)
 -- several of our implementations depend on Hadoop.

[Machine Learning Open Source Software](http://mloss.org/software/)
 -- other projects implementing Open Source Machine Learning libraries.

[Mahout -- The name, history and its pronunciation](mahoutname.html)

<a name="MahoutWiki-Community"></a>
## Community

[Who we are](who-we-are.html)
 -- who are the developers behind Apache Mahout?

[Books, Tutorials, Talks, Articles, News, Background Reading, etc. on Mahout](books-tutorials-and-talks.html)

[Issue Tracker](issue-tracker.html)
 -- see what features people are working on, submit patches and file bugs.

[Source Code (SVN)](https://svn.apache.org/repos/asf/mahout/)
 -- [Fisheye|http://fisheye6.atlassian.com/browse/mahout]
 -- download the Mahout source code from svn.

[Mailing lists and IRC](mailing-lists,-irc-and-archives.html)
 -- links to our mailing lists, IRC channel and archived design and
algorithm discussions, maybe your questions was answered there already?

[Version Control](version-control.html)
 -- where we track our code.

[Powered By Mahout](powered-by-mahout.html)
 -- who is using Mahout in production?

[Professional Support](professional-support.html)
 -- who is offering professional support for Mahout?

[Mahout and Google Summer of Code](gsoc.html)
  -- All you need to know about Mahout and GSoC.


[Glossary of commonly used terms and abbreviations](glossary.html)

<a name="MahoutWiki-Installation/Setup"></a>
## Installation/Setup

[System Requirements](system-requirements.html)
 -- what do you need to run Mahout?

[Quickstart](quickstart.html)
 -- get started with Mahout, run the examples and get pointers to further
resources.

[Downloads](downloads.html)
 -- a list of Mahout releases.

[Download and installation](buildingmahout.html)
 -- build Mahout from the sources.

[Mahout on Amazon's EC2 Service](mahout-on-amazon-ec2.html)
 -- run Mahout on Amazon's EC2.

[Mahout on Amazon's EMR](mahout-on-elastic-mapreduce.html)
 -- Run Mahout on Amazon's Elastic Map Reduce

[Integrating Mahout into an Application](mahoutintegration.html)
 -- integrate Mahout's capabilities in your application.

<a name="MahoutWiki-Examples"></a>
## Examples

1. [ASF Email Examples](asfemail.html)
 -- Examples of recommenders, clustering and classification all using a
public domain collection of 7 million emails.

<a name="MahoutWiki-ImplementationBackground"></a>
## Implementation Background

<a name="MahoutWiki-RequirementsandDesign"></a>
### Requirements and Design

[Matrix and Vector Needs](matrix-and-vector-needs.html)
 -- requirements for Mahout vectors.

[Collection(De-)Serialization](collection(de-)serialization.html)

<a name="MahoutWiki-CollectionsandAlgorithms"></a>
### Collections and Algorithms

Learn more about [mahout-collections](mahout-collections.html)
, containers for efficient storage of primitive-type data and open hash
tables.

Learn more about the [Algorithms](algorithms.html)
 discussed and employed by Mahout.

Learn more about the [Mahout recommender implementation](recommender-documentation.html)
.

<a name="MahoutWiki-Utilities"></a>
### Utilities

This section describes tools that might be useful for working with Mahout.

[Converting Content](converting-content.html)
 -- Mahout has some utilities for converting content such as logs to
formats more amenable for consumption by Mahout.
[Creating Vectors](creating-vectors.html)
 -- Mahout's algorithms operate on vectors. Learn more on how to generate
these from raw data.
[Viewing Result](viewing-result.html)
 -- How to visualize the result of your trained algorithms.

<a name="MahoutWiki-Data"></a>
### Data

[Collections](collections.html)
 -- To try out and test Mahout's algorithms you need training data. We are
always looking for new training data collections.

<a name="MahoutWiki-Benchmarks"></a>
### Benchmarks

[Mahout Benchmarks](mahout-benchmarks.html)

<a name="MahoutWiki-Committer'sResources"></a>
## Committer's Resources

* [Testing](testing.html)
 -- Information on test plans and ideas for testing

<a name="MahoutWiki-ProjectResources"></a>
### Project Resources

* [Dealing with Third Party Dependencies not in Maven](thirdparty-dependencies.html)
* [How To Update The Website](how-to-update-the-website.html)
* [Patch Check List](patch-check-list.html)
* [How To Release](http://cwiki.apache.org/confluence/display/MAHOUT/How+to+release)
* [Release Planning](release-planning.html)
* [Sonar Code Quality Analysis](https://analysis.apache.org/dashboard/index/63921)

<a name="MahoutWiki-AdditionalResources"></a>
### Additional Resources

* [Apache Machine Status](http://monitoring.apache.org/status/)
 \- Check to see if SVN, other resources are available.
* [Committer's FAQ](http://www.apache.org/dev/committers.html)
* [Apache Dev](http://www.apache.org/dev/)


<a name="MahoutWiki-HowToEditThisWiki"></a>
## How To Edit This Wiki

How to edit this Wiki

This Wiki is a collaborative site, anyone can contribute and share:

* Create an account by clicking the "Login" link at the top of any page,
and picking a username and password.
* Edit any page by pressing Edit at the top of the page

There are some conventions used on the Mahout wiki:

    * {noformat}+*TODO:*+{noformat} (+*TODO:*+ ) is used to denote sections
that definitely need to be cleaned up.
    * {noformat}+*Mahout_(version)*+{noformat} (+*Mahout_0.2*+) is used to
draw attention to which version of Mahout a feature was (or will be) added
to Mahout.

