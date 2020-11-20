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
title: Recommender Overview


---


# Recommender Overview

Recommenders have changed over the years. Mahout contains a long list of them, which you can still use. However in about 2013 there was a revolution in recommenders, which favored what we might call "Multimodal", meaning they could take in data of all sorts&mdash;basically anything we might think was an indicator of user taste. The new Samsara algorithm, called Correlated Cross-Occurrence (CCO) is just such a next gen recommender algorithm but Mahout-Samsara only implements the model building part. This can be integrated as the user see fit and the rest of this doc will explain how.

## Turnkey Implementation

If you are looking for an end-to-end OSS recommender based on the Mahout CCO algorithm have a look at [The Universal Recommender](https://github.com/actionml/universal-recommender), which is implemented using [Apache PredictionIO](http://predictionio.apache.org/). See instructions for [installation here](http://actionml.com/docs/pio_by_actionml). There is even an AWS AMI for convenience (this is a for-pay option)

## Build Your Own Integration

To get the most out of our more modern CCO algorithm we'll need to think of the Recommender as a "model creation" component&mdash;supplied by Mahout's new spark-itemsimilarity job, and a "serving" component&mdash;supplied by a modern scalable search engine, like Solr or Elasticsearch. Here we describe a loose integration that does not require using Mahout as a library, it uses Mahout's command line interface. This is clearly not the best but allows one to experiments and get a real recommender running easily.

![image](http://i.imgur.com/fliHMBo.png)

To integrate with your application you will collect user interactions storing them in a DB and also in a from usable by Mahout. The simplest way to do this is to log user interactions to csv files (user-id, item-id). The DB should be setup to contain the last n user interactions, which will form part of the query for recommendations.

Mahout's spark-itemsimilarity will create a table of (item-id, list-of-similar-items) in csv form. Think of this as an item collection with one field containing the item-ids of similar items. Index this with your search engine.

When your application needs recommendations for a specific person, get the latest user history of interactions from the DB and query the indicator collection with this history. You will get back an ordered list of item-ids. These are your recommendations. You may wish to filter out any that the user has already seen but that will depend on your use case.

All ids for users and items are preserved as string tokens and so work as an external key in DBs or as doc ids for search engines, they also work as tokens for search queries.

## References

1. A free ebook, which talks about the general idea: [Practical Machine Learning](https://www.mapr.com/practical-machine-learning)
2. A slide deck, which talks about mixing actions or other indicators: [Creating a Multimodal Recommender with Mahout and a Search Engine](http://occamsmachete.com/ml/2014/10/07/creating-a-unified-recommender-with-mahout-and-a-search-engine/)
3. Two blog posts: [What's New in Recommenders: part #1](http://occamsmachete.com/ml/2014/08/11/mahout-on-spark-whats-new-in-recommenders/)
and  [What's New in Recommenders: part #2](http://occamsmachete.com/ml/2014/09/09/mahout-on-spark-whats-new-in-recommenders-part-2/)
3. A post describing the loglikelihood ratio:  [Surprise and Coinsidense](http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html)  LLR is used to reduce noise in the data while keeping the calculations O(n) complexity.

## Mahout Model Creation

See the page describing [*spark-itemsimilarity*](http://mahout.apache.org/users/recommender/intro-cooccurrence-spark.html) for more details.
