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


# Mahout Quickstart Guide

## Getting Mahout

### Download the Latest Release

You can download the latest official Mahout release from the Apache downloads page:

👉 **https://downloads.apache.org/mahout/**

Or access the latest source code on GitHub:

👉 **https://github.com/apache/mahout.git**

### Add Mahout to a Maven Project

Mahout is available on Maven Central under the group ID `org.apache.mahout`.

To use the latest stable release, add the following to your **pom.xml**:

```xml
<dependency>
    <groupId>org.apache.mahout</groupId>
    <artifactId>mahout-mr</artifactId>
    <version>14.1</version> 
</dependency>
```

> Note: Although MapReduce components are still available, Mahout now primarily focuses on linear algebra and Samsara DSL. See the documentation for current engine support.

## Features

For a full list of Mahout’s features, visit:

👉 **https://mahout.apache.org/community/mailing-lists.html**

## Using Mahout

Mahout provides examples and tutorials to help users explore its algorithms.

### Recommendations

- [Recommender Quickstart](/users/recommender/quickstart.html)  
- [User-Based Recommender in 5 Minutes](/users/recommender/userbased-5-minutes.html)  
- [Recommender Dos and Don’ts](/users/recommender/recommender-first-timer-faq.html)

### Clustering

- [Clustering Synthetic Data](/users/clustering/clustering-of-synthetic-control-data.html)

### Classification

- [Naive Bayes on 20 Newsgroups](/users/classification/twenty-newsgroups.html)  
- [Hidden Markov Models Example](/users/classification/hidden-markov-models.html)  
- [Random Forest Quickstart](/users/classification/partial-implementation.html)

### Working with Text

To convert raw text into vectors:

- [Creating Vectors from Text](/users/basics/creating-vectors-from-text.html). 