---
layout: post
title:  "Apache Mahout 0.13.1 Release"
date:   2017-04-17 18:40:52 -0400
categories:
  - new release
  - update
image: https://source.unsplash.com/RkJF2BMrLJc/100x100
---

Mahout 0.13.0 is here! The new release includes the following additions and changes:

-   In-core matrices backed by ViennaCL 3 providing in some cases speedups of an order of magnitude.
-   A JavaCPP bridge to native/GPU operations in ViennaCL
-   Distributed GPU Matrix-Matrix and Matrix-Vector multiplication on Spark
-   Distributed OpenMP Matrix-Matrix and Matrix-Vector multiplication on Spark
-   Sparse and dense matrix GPU-backed support.
-   Fault tolerance by falling back to Mahout JVM counterpart of new solvers in the case of failure on GPU or OpenMP
-   A new scikit-learn-like framework for algorithms with the goal for creating a consistent API for various machine-learning algorithms and an orderly package structure for grouping regression, classification, clustering, and pre-processing algorithms together
-   New DRM wrappers in Spark Bindings making it more convenient to create DRMs from MLLib RDDs and DataFrames
-   MahoutConversions adds Scala-like compatibility to Vectors introducing methods such as toArray() and toMap()
