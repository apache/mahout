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

#Introduction

This document provides an overview of how the Mahout Samsara environment is implemented over the Apache Flink backend engine. This document gives an overview of the code layout for the Flink backend engine, the source code for which can be found under /flink directory in the Mahout codebase.

Apache Flink is a distributed big data streaming engine that supports both Streaming and Batch interfaces. Batch processing is an extension of Flink’s Stream processing engine.

The Mahout Flink integration presently supports Flink’s batch processing capabilities leveraging the DataSet API.

The Mahout DRM, or Distributed Row Matrix, is an abstraction for storing a large matrix of numbers in-memory in a cluster by distributing logical rows among servers. Mahout's scala DSL provides an abstract API on DRMs for backend engines to provide implementations of this API. An example is the Spark backend engine. Each engine has it's own design of mapping the abstract API onto its data model and provides implementations for algebraic operators over that mapping.

#Flink Overview

Apache Flink is an open source, distributed Stream and Batch Processing Framework. At it's core, Flink is a Stream Processing engine and Batch processing is an extension of Stream Processing. 

Flink includes several APIs for building applications with the Flink Engine:

 <ol>
<li><b>DataSet API</b> for Batch data in Java, Scala and Python</li>
<li><b>DataStream API</b> for Stream Processing in Java and Scala</li>
<li><b>Table API</b> with SQL-like regular expression language in Java and Scala</li>
<li><b>Gelly</b> Graph Processing API in Java and Scala</li>
<li><b>CEP API</b>, a complex event processing library</li>
<li><b>FlinkML</b>, a Machine Learning library</li>
</ol>
#Flink Environment Engine

The Flink backend implements the abstract DRM as a Flink DataSet. A Flink job runs in the context of an ExecutionEnvironment (from the Flink Batch processing API).

#Source Layout

Within mahout.git, the top level directory, flink/ holds all the source code for the Flink backend engine. Sections of code that interface with the rest of the Mahout components are in Scala, and sections of the code that interface with Flink DataSet API and implement algebraic operators are in Java. Here is a brief overview of what functionality can be found within flink/ folder.

flink/ - top level directory containing all Flink related code

flink/src/main/scala/org/apache/mahout/flinkbindings/blas/*.scala - Physical operator code for the Samsara DSL algebra

flink/src/main/scala/org/apache/mahout/flinkbindings/drm/*.scala - Flink Dataset DRM and broadcast implementation

flink/src/main/scala/org/apache/mahout/flinkbindings/io/*.scala - Read / Write between DRMDataSet and files on HDFS

flink/src/main/scala/org/apache/mahout/flinkbindings/FlinkEngine.scala - DSL operator graph evaluator and various abstract API implementations for a distributed engine.


