---
layout: default
title: Recommender Quickstart
theme:
    name: retro-mahout
---

# Recommender Overview

Recommenders have changed over the years. Mahout contains a long list of them, which you can still use. But to get the best  out of our more modern aproach we'll need to think of the Recommender as a "model creation" component&mdash;supplied by Mahout's new spark-itemsimilarity job, and a "serving" component&mdash;supplied by a modern scalable search engine, like Solr.

![image](http://i.imgur.com/fliHMBo.png)

To integrate with your application you will collect user interactions storing them in a DB and also in a from usable by Mahout. The simplest way to do this is to log user interactions to csv files (user-id, item-id). The DB should be setup to contain the last n user interactions, which will form part of the query for recommendations.

Mahout's spark-itemsimilarity will create a table of (item-id, list-of-similar-items) in csv form. Think of this as an item collection with one field containing the item-ids of similar items. Index this with your search engine. 

When your application needs recommendations for a specific person, get the latest user history of interactions from the DB and query the indicator collection with this history. You will get back an ordered list of item-ids. These are your recommendations. You may wish to filter out any that the user has already seen but that will depend on your use case.

All ids for users and items are preserved as string tokens and so work as an external key in DBs or as doc ids for search engines, they also work as tokens for search queries.

##References

1. A free ebook, which talks about the general idea: [Practical Machine Learning](https://www.mapr.com/practical-machine-learning)
2. A slide deck, which talks about mixing actions or other indicators: [Creating a Multimodal Recommender with Mahout and a Search Engine](http://occamsmachete.com/ml/2014/10/07/creating-a-unified-recommender-with-mahout-and-a-search-engine/)
3. Two blog posts: [What's New in Recommenders: part #1](http://occamsmachete.com/ml/2014/08/11/mahout-on-spark-whats-new-in-recommenders/)
and  [What's New in Recommenders: part #2](http://occamsmachete.com/ml/2014/09/09/mahout-on-spark-whats-new-in-recommenders-part-2/)
3. A post describing the loglikelihood ratio:  [Surprise and Coinsidense](http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html)  LLR is used to reduce noise in the data while keeping the calculations O(n) complexity.

##Mahout Model Creation

See the page describing [*spark-itemsimilarity*](http://mahout.apache.org/users/recommender/intro-cooccurrence-spark.html) for more details.