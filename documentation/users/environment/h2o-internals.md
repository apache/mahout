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
title: 

   
---

# Introduction

This document provides an overview of how the Mahout Samsara environment is implemented over the H2O backend engine. The document is aimed at Mahout developers, to give a high level description of the design so that one can explore the code inside `h2o/` with some context.

## H2O Overview

H2O is a distributed scalable machine learning system. Internal architecture of H2O has a distributed math engine (h2o-core) and a separate layer on top for algorithms and UI. The Mahout integration requires only the math engine (h2o-core).

## H2O Data Model

The data model of the H2O math engine is a distributed columnar store (of primarily numbers, but also strings). A column of numbers is called a Vector, which is broken into Chunks (of a few thousand elements). Chunks are distributed across the cluster based on a deterministic hash. Therefore, any member of the cluster knows where a particular Chunk of a Vector is homed. Each Chunk is separately compressed in memory and elements are individually decompressed on the fly upon access with purely register operations (thereby achieving high memory throughput). An ordered set of similarly partitioned Vecs are composed into a Frame. A Frame is therefore a large two dimensional table of numbers. All elements of a logical row in the Frame are guaranteed to be homed in the same server of the cluster. Generally speaking, H2O works well on "tall skinny" data, i.e, lots of rows (100s of millions) and modest number of columns (10s of thousands).


## Mahout DRM

The Mahout DRM, or Distributed Row Matrix, is an abstraction for storing a large matrix of numbers in-memory in a cluster by distributing logical rows among servers. Mahout's scala DSL provides an abstract API on DRMs for backend engines to provide implementations of this API. Examples are the Spark and H2O backend engines. Each engine has it's own design of mapping the abstract API onto its data model and provides implementations for algebraic operators over that mapping.


## H2O Environment Engine

The H2O backend implements the abstract DRM as an H2O Frame. Each logical column in the DRM is an H2O Vector. All elements of a logical DRM row are guaranteed to be homed on the same server. A set of rows stored on a server are presented as a read-only virtual in-core Matrix (i.e BlockMatrix) for the closure method in the `mapBlock(...)` API.

H2O provides a flexible execution framework called `MRTask`. The `MRTask` framework typically executes over a Frame (or even a Vector), supports various types of map() methods, can optionally modify the Frame or Vector (though this never happens in the Mahout integration), and optionally create a new Vector or set of Vectors (to combine them into a new Frame, and consequently a new DRM).


## Source Layout

Within mahout.git, the top level directory, `h2o/` holds all the source code related to the H2O backend engine. Part of the code (that interfaces with the rest of the Mahout componenets) is in Scala, and part of the code (that interfaces with h2o-core and implements algebraic operators) is in Java. Here is a brief overview of what functionality can be found where within `h2o/`.

  h2o/ - top level directory containing all H2O related code

  h2o/src/main/java/org/apache/mahout/h2obindings/ops/*.java - Physical operator code for the various DSL algebra

  h2o/src/main/java/org/apache/mahout/h2obindings/drm/*.java - DRM backing (onto Frame) and Broadcast implementation

  h2o/src/main/java/org/apache/mahout/h2obindings/H2OHdfs.java - Read / Write between DRM (Frame) and files on HDFS

  h2o/src/main/java/org/apache/mahout/h2obindings/H2OBlockMatrix.java - A vertical block matrix of DRM presented as a virtual copy-on-write in-core Matrix. Used in mapBlock() API

  h2o/src/main/java/org/apache/mahout/h2obindings/H2OHelper.java - A collection of various functionality and helpers. For e.g, convert between in-core Matrix and DRM, various summary statistics on DRM/Frame.

  h2o/src/main/scala/org/apache/mahout/h2obindings/H2OEngine.scala - DSL operator graph evaluator and various abstract API implementations for a distributed engine

  h2o/src/main/scala/org/apache/mahout/h2obindings/* - Various abstract API implementations ("glue work")