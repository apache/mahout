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
title: FAQ

    
---

# The Official Mahout FAQ

*General*

1. [What is Apache Mahout?](#whatis)
1. [What does the name mean?](#mean)
1. [How is the name pronounced?](#pronounce)
1. [Where can I find the origins of the Mahout project?](#historical)
1. [Where can I download the Mahout logo?](#downloadlogo)
1. [Where can I download Mahout slide presentations?](#presentations)

*Algorithms*

1. [What algorithms are implemented in Mahout?](#algos)
1. [What algorithms are missing from Mahout?](#todo)
1. [Do I need Hadoop to run Mahout?](#hadoop)

*Hadoop specific questions*

1. [Mahout just won't run in parallel on my dataset. Why?](#split)


# *Answers*


## General


<a name="whatis"></a>
#### What is Apache Mahout?

Apache Mahout is a suite of machine learning libraries designed to be
scalable and robust

<a name="mean"></a>
#### What does the name mean?

The name [Mahout](http://en.wikipedia.org/wiki/Mahout)
 was original chosen for it's association with the [Apache Hadoop](http://hadoop.apache.org)
 project.  A Mahout is a person who drives an elephant (hint: Hadoop's logo
is an elephant).  We just wanted a name that complemented Hadoop but we see
our project as a good driver of Hadoop in the sense that we will be using
and testing it.  We are not, however, implying that we are controlling
Hadoop's development.

Prior to coming to the ASF, those of us working on the project plan voted between [Howdah](http://en.wikipedia.org/wiki/Howdah) – the carriage on top of an elephant and Mahout.

<a name="historical"></a>
#### Where can I find the origins of the Mahout project?

See [http://ml-site.grantingersoll.com](http://web.archive.org/web/20080101233917/http://ml-site.grantingersoll.com/index.php?title=Main_Page)
 for old wiki and mailing list archives (all read-only)

Mahout was started by <a href="http://web.archive.org/web/20071228055210/http://ml-site.grantingersoll.com/index.php?title=Main_Page" class="external-link" rel="nofollow">Isabel Drost, Grant Ingersoll and Karl Wettin</a>. It <a href="http://web.archive.org/web/20080201093120/http://lucene.apache.org/#22+January+2008+-+Lucene+PMC+Approves+Mahout+Machine+Learning+Project" class="external-link" rel="nofollow">started</a> as part of the <a href="http://lucene.apache.org" class="external-link" rel="nofollow">Lucene</a> project (see the <a href="http://web.archive.org/web/20080102151102/http://ml-site.grantingersoll.com/index.php?title=Incubator_proposal" class="external-link" rel="nofollow">original proposal</a>) and went on to become a top level project in April of 2010.</p><p style="text-align: left;">The original goal was to implement all 10 algorithms from Andrew Ng's paper &quot;<a href="http://ai.stanford.edu/~ang/papers/nips06-mapreducemulticore.pdf" class="external-link" rel="nofollow">Map-Reduce for Machine Learning on Multicore</a>&quot;</p>

<a name="pronounce"></a>
#### How is the name pronounced?

There are some disagreements about how to pronounce the name. Webster's has it as muh-hout (as in ["out"](http://dictionary.reference.com/browse/mahout)), but the Sanskrit/Hindi origins pronounce it as "muh-hoot". The second pronunciation suggests a nice pun on the Hebrew word מהות meaning "essence or truth".

<a name="downloadlogo"></a>
#### Where can I download the Mahout logo?

See [MAHOUT-335](https://issues.apache.org/jira/browse/MAHOUT-335)


<a name="presentations"></a>
#### Where can I download Mahout slide presentations?

The [Books, Tutorials and Talks](https://mahout.apache.org/general/books-tutorials-and-talks.html)
 page contains an overview of a wide variety of presentations with links to slides where available.

## Algorithms

<a name="algos"></a>
#### What algorithms are implemented in Mahout?

We are interested in a wide variety of machine learning algorithms. Many of
which are already implemented in Mahout. You can find a list [here](https://mahout.apache.org/users/basics/algorithms.html).

<a name="todo"></a>
#### What algorithms are missing from Mahout?

There are many machine learning algorithms that we would like to have in
Mahout. If you have an algorithm or an improvement to an algorithm that you would
like to implement, start a discussion on our [mailing list](https://mahout.apache.org/general/mailing-lists,-irc-and-archives.html).

<a name="hadoop"></a>
#### Do I need Hadoop to use Mahout?

There is a number of algorithm implementations that require no Hadoop dependencies whatsoever, consult the [algorithms list](https://mahout.apache.org/users/basics/algorithms.html). In the future, we might provide more algorithm implementations on platforms more suitable for machine learning such as [Apache Spark](http://spark.apache.org)

## Hadoop specific questions
<a name="split"></a>
#### Mahout just won't run in parallel on my dataset. Why?

If you are running training on a Hadoop cluster keep in mind that the number of mappers started is governed by the size of the input data and the configured split/block size of your cluster. As a rule of thumb,
anything below 100MB in size won't be split by default. 