---
layout: default
title: Downloads
theme: mahout
---
---
layout: default
title: Overview
<a name="Overview-OverviewofMahout"></a>
# Overview of Mahout

Mahout's goal is to build scalable machine learning libraries. With
scalable we mean: 
* Scalable to reasonably large data sets. Our core algorithms for
clustering, classification and batch based collaborative filtering are
implemented on top of Apache Hadoop using the map/reduce paradigm. However
we do not restrict contributions to Hadoop based implementations:
Contributions that run on a single node or on a non-Hadoop cluster are
welcome as well. The core libraries are highly optimized to allow for good
performance also for non-distributed algorithms.
* Scalable to support your business case. Mahout is distributed under a
commercially friendly Apache Software license.
* Scalable community. The goal of Mahout is to build a vibrant, responsive,
diverse community to facilitate discussions not only on the project itself
but also on potential use cases. Come to the mailing lists to find out
more.


Currently Mahout supports mainly four use cases: Recommendation mining
takes users' behavior and from that tries to find items users might like.
Clustering takes e.g. text documents and groups them into groups of
topically related documents. Classification learns from exisiting
categorized documents what documents of a specific category look like and
is able to assign unlabelled documents to the (hopefully) correct category.
Frequent itemset mining takes a set of item groups (terms in a query
session, shopping cart content) and identifies, which individual items
usually appear together. 

Interested in helping? See the [Wiki](http://cwiki.apache.org/confluence/display/MAHOUT)
 or send us an email. Also note, we are just getting off the ground, so
please be patient as we get the various infrastructure pieces in place.
