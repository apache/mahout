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
layout: doc-page
title: Quickstart

    
---
# Mahout Quick Start 
# TODO : Fill this in with the bare essential basics



# Mahout MapReduce Overview

## Getting Mahout

#### Download the latest release

Download the latest release [here](http://www.apache.org/dyn/closer.cgi/mahout/).

Or checkout the latest code from [here](http://mahout.apache.org/developers/version-control.html)

#### Alternatively: Add Mahout 0.13.0 to a maven project

Mahout is also available via a [maven repository](http://mvnrepository.com/artifact/org.apache.mahout) under the group id *org.apache.mahout*.
If you would like to import the latest release of mahout into a java project, add the following dependency in your *pom.xml*:

    <dependency>
        <groupId>org.apache.mahout</groupId>
        <artifactId>mahout-mr</artifactId>
        <version>0.14.0</version>
    </dependency>
 

## Features

For a full list of Mahout's features see our [Features by Engine](http://mahout.apache.org/users/basics/algorithms.html) page.

    
## Using Mahout

Mahout has prepared a bunch of examples and tutorials for users to quickly learn how to use its machine learning algorithms.

#### Recommendations

Check the [Recommender Quickstart](/users/recommender/quickstart.html) or the tutorial on [creating a userbased recommender in 5 minutes](/users/recommender/userbased-5-minutes.html).

If you are building a recommender system for the first time, please also refer to a list of [Dos and Don'ts](/users/recommender/recommender-first-timer-faq.html) that might be helpful.

#### Clustering

Check the [Synthetic data](/users/clustering/clustering-of-synthetic-control-data.html) example.

#### Classification

If you are interested in how to train a **Naive Bayes** model, look at the [20 newsgroups](/users/classification/twenty-newsgroups.html) example.

If you plan to build a **Hidden Markov Model** for speech recognition, the example [here](/users/classification/hidden-markov-models.html) might be instructive. 

Or you could build a **Random Forest** model by following this [quick start page](/users/classification/partial-implementation.html).

#### Working with Text 

If you need to convert raw text into word vectors as input to clustering or classification algorithms, please refer to this page on [how to create vectors from text](/users/basics/creating-vectors-from-text.html).
