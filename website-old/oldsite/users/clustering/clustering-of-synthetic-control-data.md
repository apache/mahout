---
layout: default
title: Clustering of synthetic control data
theme:
   name: retro-mahout
---

# Clustering synthetic control data

## Introduction

This example will demonstrate clustering of time series data, specifically control charts. [Control charts](http://en.wikipedia.org/wiki/Control_chart) are tools used to determine whether a manufacturing or business process is in a state of statistical control. Such control charts are generated / simulated repeatedly at equal time intervals. A [simulated dataset](http://archive.ics.uci.edu/ml/databases/synthetic_control/synthetic_control.data.html) is available for use in UCI machine learning repository.

A time series of control charts needs to be clustered into their close knit groups. The data set we use is synthetic and is meant to resemble real world information in an anonymized format. It contains six different classes: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift. In this example we will use Mahout to cluster the data into corresponding class buckets. 

*For the sake of simplicity, we won't use a cluster in this example, but instead show you the commands to run the clustering examples locally with Hadoop*.

## Setup

We need to do some initial setup before we are able to run the example. 


  1. Start out by downloading the dataset to be clustered from the UCI Machine Learning Repository: [http://archive.ics.uci.edu/ml/databases/synthetic_control/synthetic_control.data](http://archive.ics.uci.edu/ml/databases/synthetic_control/synthetic_control.data).

  2. Download the [latest release of Mahout](/general/downloads.html).

  3. Unpack the release binary and switch to the *mahout-distribution-0.x* folder

  4. Make sure that the *JAVA_HOME* environment variable points to your local java installation

  5. Create a folder called *testdata* in the current directory and copy the dataset into this folder.


## Clustering Examples

Depending on the clustering algorithm you want to run, the following commands can be used:


   * [Canopy Clustering](/users/clustering/canopy-clustering.html)

    bin/mahout org.apache.mahout.clustering.syntheticcontrol.canopy.Job

   * [k-Means Clustering](/users/clustering/k-means-clustering.html)

    bin/mahout org.apache.mahout.clustering.syntheticcontrol.kmeans.Job


   * [Fuzzy k-Means Clustering](/users/clustering/fuzzy-k-means.html)

    bin/mahout org.apache.mahout.clustering.syntheticcontrol.fuzzykmeans.Job

The clustering output will be produced in the *output* directory. The output data points are in vector format. In order to read/analyze the output, you can use the [clusterdump](/users/clustering/cluster-dumper.html) utility provided by Mahout.

