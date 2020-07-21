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
title: 

   
---

## Getting Started 

To get started, add the following dependency to the pom:

    <dependency>
      <groupId>org.apache.mahout</groupId>
      <artifactId>mahout-flink_2.10</artifactId>
      <version>0.12.0</version>
    </dependency>

Here is how to use the Flink backend:

	import org.apache.flink.api.scala._
	import org.apache.mahout.math.drm._
	import org.apache.mahout.math.drm.RLikeDrmOps._
	import org.apache.mahout.flinkbindings._

	object ReadCsvExample {

	  def main(args: Array[String]): Unit = {
	    val filePath = "path/to/the/input/file"

	    val env = ExecutionEnvironment.getExecutionEnvironment
	    implicit val ctx = new FlinkDistributedContext(env)

	    val drm = readCsv(filePath, delim = "\t", comment = "#")
	    val C = drm.t %*% drm
	    println(C.collect)
	  }

	}

## Current Status

The top JIRA for Flink backend is [MAHOUT-1570](https://issues.apache.org/jira/browse/MAHOUT-1570) which has been fully implemented.

### Implemented

* [MAHOUT-1701](https://issues.apache.org/jira/browse/MAHOUT-1701) Mahout DSL for Flink: implement AtB ABt and AtA operators
* [MAHOUT-1702](https://issues.apache.org/jira/browse/MAHOUT-1702) implement element-wise operators (like `A + 2` or `A + B`) 
* [MAHOUT-1703](https://issues.apache.org/jira/browse/MAHOUT-1703) implement `cbind` and `rbind`
* [MAHOUT-1709](https://issues.apache.org/jira/browse/MAHOUT-1709) implement slicing (like `A(1 to 10, ::)`)
* [MAHOUT-1710](https://issues.apache.org/jira/browse/MAHOUT-1710) implement right in-core matrix multiplication (`A %*% B` when `B` is in-core) 
* [MAHOUT-1711](https://issues.apache.org/jira/browse/MAHOUT-1711) implement broadcasting
* [MAHOUT-1712](https://issues.apache.org/jira/browse/MAHOUT-1712) implement operators `At`, `Ax`, `Atx` - `Ax` and `At` are implemented
* [MAHOUT-1734](https://issues.apache.org/jira/browse/MAHOUT-1734) implement I/O - should be able to read results of Flink bindings
* [MAHOUT-1747](https://issues.apache.org/jira/browse/MAHOUT-1747) add support for different types of indexes (String, long, etc) - now supports `Int`, `Long` and `String`
* [MAHOUT-1748](https://issues.apache.org/jira/browse/MAHOUT-1748) switch to Flink Scala API 
* [MAHOUT-1749](https://issues.apache.org/jira/browse/MAHOUT-1749) Implement `Atx`
* [MAHOUT-1750](https://issues.apache.org/jira/browse/MAHOUT-1750) Implement `ABt`
* [MAHOUT-1751](https://issues.apache.org/jira/browse/MAHOUT-1751) Implement `AtA` 
* [MAHOUT-1755](https://issues.apache.org/jira/browse/MAHOUT-1755) Flush intermediate results to FS - Flink, unlike Spark, does not store intermediate results in memory.
* [MAHOUT-1764](https://issues.apache.org/jira/browse/MAHOUT-1764) Add standard backend tests for Flink
* [MAHOUT-1765](https://issues.apache.org/jira/browse/MAHOUT-1765) Add documentation about Flink backend
* [MAHOUT-1776](https://issues.apache.org/jira/browse/MAHOUT-1776) Refactor common Engine agnostic classes to Math-Scala module
* [MAHOUT-1777](https://issues.apache.org/jira/browse/MAHOUT-1777) move HDFSUtil classes into the HDFS module
* [MAHOUT-1804](https://issues.apache.org/jira/browse/MAHOUT-1804) Implement drmParallelizeWithRowLabels(..) in Flink
* [MAHOUT-1805](https://issues.apache.org/jira/browse/MAHOUT-1805) Implement allReduceBlock(..) in Flink bindings
* [MAHOUT-1809](https://issues.apache.org/jira/browse/MAHOUT-1809) Failing tests in flin-bindings: dals and dspca
* [MAHOUT-1810](https://issues.apache.org/jira/browse/MAHOUT-1810) Failing test in flink-bindings: A + B Identically partitioned (mapBlock Checkpointing issue)
* [MAHOUT-1812](https://issues.apache.org/jira/browse/MAHOUT-1812) Implement drmParallelizeWithEmptyLong(..) in flink bindings
* [MAHOUT-1814](https://issues.apache.org/jira/browse/MAHOUT-1814) Implement drm2intKeyed in flink bindings
* [MAHOUT-1815](https://issues.apache.org/jira/browse/MAHOUT-1815) dsqDist(X,Y) and dsqDist(X) failing in flink tests
* [MAHOUT-1816](https://issues.apache.org/jira/browse/MAHOUT-1816) Implement newRowCardinality in CheckpointedFlinkDrm
* [MAHOUT-1817](https://issues.apache.org/jira/browse/MAHOUT-1817) Implement caching in Flink Bindings
* [MAHOUT-1818](https://issues.apache.org/jira/browse/MAHOUT-1818) dals test failing in Flink Bindings
* [MAHOUT-1819](https://issues.apache.org/jira/browse/MAHOUT-1819) Set the default Parallelism for Flink execution in FlinkDistributedContext
* [MAHOUT-1820](https://issues.apache.org/jira/browse/MAHOUT-1820) Add a method to generate Tuple<PartitionId, Partition elements count>> to support Flink backend
* [MAHOUT-1821](https://issues.apache.org/jira/browse/MAHOUT-1821) Use a mahout-flink-conf.yaml configuration file for Mahout specific Flink configuration
* [MAHOUT-1822](https://issues.apache.org/jira/browse/MAHOUT-1822) Update NOTICE.txt, License.txt to add Apache Flink
* [MAHOUT-1823](https://issues.apache.org/jira/browse/MAHOUT-1823) Modify MahoutFlinkTestSuite to implement FlinkTestBase
* [MAHOUT-1824](https://issues.apache.org/jira/browse/MAHOUT-1824) Optimize FlinkOpAtA to use upper triangular matrices
* [MAHOUT-1825](https://issues.apache.org/jira/browse/MAHOUT-1825) Add List of Flink algorithms to Mahout wiki page

### Tests 

There is a set of standard tests that all engines should pass (see [MAHOUT-1764](https://issues.apache.org/jira/browse/MAHOUT-1764)).  

* `DistributedDecompositionsSuite` 
* `DrmLikeOpsSuite` 
* `DrmLikeSuite` 
* `RLikeDrmOpsSuite` 


These are Flink-backend specific tests, e.g.

* `DrmLikeOpsSuite` for operations like `norm`, `rowSums`, `rowMeans`
* `RLikeOpsSuite` for basic LA like `A.t %*% A`, `A.t %*% x`, etc
* `LATestSuite` tests for specific operators like `AtB`, `Ax`, etc
* `UseCasesSuite` has more complex examples, like power iteration, ridge regression, etc

## Environment 

For development the minimal supported configuration is 

* [JDK 1.7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html)
* [Scala 2.10]

When using mahout, please import the following modules: 

* `mahout-math`
* `mahout-math-scala`
* `mahout-flink_2.10`
*