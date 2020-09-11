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
title: NaiveBayes

    
---

<a name="NaiveBayes-NaiveBayes"></a>
# Naive Bayes

Naive Bayes is an algorithm that can be used to classify objects into
usually binary categories. It is one of the most common learning algorithms
in spam filters. Despite its simplicity and rather naive assumptions it has
proven to work surprisingly well in practice.

Before applying the algorithm, the objects to be classified need to be
represented by numerical features. In the case of e-mail spam each feature
might indicate whether some specific word is present or absent in the mail
to classify. The algorithm comes in two phases: Learning and application.
During learning, a set of feature vectors is given to the algorithm, each
vector labeled with the class the object it represents, belongs to. From
that it is deduced which combination of features appears with high
probability in spam messages. Given this information, during application
one can easily compute the probability of a new message being either spam
or not.

The algorithm does make several assumptions, that are not true for most
datasets, but make computations easier. The worst probably being, that all
features of an objects are considered independent. In practice, that means,
given the phrase "Statue of Liberty" was already found in a text, does not
influence the probability of seeing the phrase "New York" as well.

<a name="NaiveBayes-StrategyforaparallelNaiveBayes"></a>
## Strategy for a parallel Naive Bayes

See [https://issues.apache.org/jira/browse/MAHOUT-9](https://issues.apache.org/jira/browse/MAHOUT-9)
.


<a name="NaiveBayes-Examples"></a>
## Examples

[20Newsgroups](20newsgroups.html)
 - Example code showing how to train and use the Naive Bayes classifier
using the 20 Newsgroups data available at [http://people.csail.mit.edu/jrennie/20Newsgroups/]
