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
title: Spark Bindings

    
---

# Scala & Spark Bindings:
*Bringing algebraic semantics*

## What is Scala & Spark Bindings?

In short, Scala & Spark Bindings for Mahout is Scala DSL and algebraic optimizer of something like this (actual formula from **(d)spca**)
        

`\[\mathbf{G}=\mathbf{B}\mathbf{B}^{\top}-\mathbf{C}-\mathbf{C}^{\top}+\mathbf{s}_{q}\mathbf{s}_{q}^{\top}\boldsymbol{\xi}^{\top}\boldsymbol{\xi}\]`

bound to in-core and distributed computations (currently, on Apache Spark).


Mahout Scala & Spark Bindings expression of the above:

        val g = bt.t %*% bt - c - c.t + (s_q cross s_q) * (xi dot xi)

The main idea is that a scientist writing algebraic expressions cannot care less of distributed 
operation plans and works **entirely on the logical level** just like he or she would do with R.

Another idea is decoupling logical expression from distributed back-end. As more back-ends are added, 
this implies **"write once, run everywhere"**.

The linear algebra side works with scalars, in-core vectors and matrices, and Mahout Distributed
Row Matrices (DRMs).

The ecosystem of operators is built in the R's image, i.e. it follows R naming such as %*%, 
colSums, nrow, length operating over vectors or matices. 

Important part of Spark Bindings is expression optimizer. It looks at expression as a whole 
and figures out how it can be simplified, and which physical operators should be picked. For example,
there are currently about 5 different physical operators performing DRM-DRM multiplication
picked based on matrix geometry, distributed dataset partitioning, orientation etc. 
If we count in DRM by in-core combinations, that would be another 4, i.e. 9 total -- all of it for just 
simple x %*% y logical notation.

Please refer to the documentation for details.

## Status

This environment addresses mostly R-like Linear Algebra optmizations for 
Spark, Flink and H20.


## Documentation

* Scala and Spark bindings manual: [web](http://apache.github.io/mahout/doc/ScalaSparkBindings.html), [pdf](ScalaSparkBindings.pdf), [pptx](MahoutScalaAndSparkBindings.pptx)
* [Spark Bindings FAQ](faq.html)
<!-- dead link* Overview blog on 0.10.x releases: [blog](http://www.weatheringthroughtechdays.com/2015/04/mahout-010x-first-mahout-release-as.html) -->

## Distributed methods and solvers using Bindings

* In-core ([ssvd]) and Distributed ([dssvd]) Stochastic SVD -- guinea pigs -- see the bindings manual
* In-core ([spca]) and Distributed ([dspca]) Stochastic PCA -- guinea pigs -- see the bindings manual
* Distributed thin QR decomposition ([dqrThin]) -- guinea pig -- see the bindings manual 
* [Current list of algorithms](https://mahout.apache.org/users/basics/algorithms.html)

[ssvd]: https://github.com/apache/mahout/blob/trunk/math-scala/src/main/scala/org/apache/mahout/math/scalabindings/SSVD.scala
[spca]: https://github.com/apache/mahout/blob/trunk/math-scala/src/main/scala/org/apache/mahout/math/scalabindings/SSVD.scala
[dssvd]: https://github.com/apache/mahout/blob/trunk/spark/src/main/scala/org/apache/mahout/sparkbindings/decompositions/DSSVD.scala
[dspca]: https://github.com/apache/mahout/blob/trunk/spark/src/main/scala/org/apache/mahout/sparkbindings/decompositions/DSPCA.scala
[dqrThin]: https://github.com/apache/mahout/blob/trunk/spark/src/main/scala/org/apache/mahout/sparkbindings/decompositions/DQR.scala

## Reading RDDs and DataFrames into DRMs
TODO


TODO: Do we still want this? (I don't think so...)
## Related history of note 

* CLI and Driver for Spark version of item similarity -- [MAHOUT-1541](https://issues.apache.org/jira/browse/MAHOUT-1541)
* Command line interface for generalizable Spark pipelines -- [MAHOUT-1569](https://issues.apache.org/jira/browse/MAHOUT-1569)
* Cooccurrence Analysis / Item-based Recommendation -- [MAHOUT-1464](https://issues.apache.org/jira/browse/MAHOUT-1464)
* Spark Bindings -- [MAHOUT-1346](https://issues.apache.org/jira/browse/MAHOUT-1346)
* Scala Bindings -- [MAHOUT-1297](https://issues.apache.org/jira/browse/MAHOUT-1297)
* Interactive Scala & Spark Bindings Shell & Script processor -- [MAHOUT-1489](https://issues.apache.org/jira/browse/MAHOUT-1489)
* OLS tutorial using Mahout shell -- [MAHOUT-1542](https://issues.apache.org/jira/browse/MAHOUT-1542)
* Full abstraction of DRM apis and algorithms from a distributed engine -- [MAHOUT-1529](https://issues.apache.org/jira/browse/MAHOUT-1529)
* Port Naive Bayes -- [MAHOUT-1493](https://issues.apache.org/jira/browse/MAHOUT-1493)

## Work in progress 
* Text-delimited files for input and output -- [MAHOUT-1568](https://issues.apache.org/jira/browse/MAHOUT-1568)
<!-- * Weighted (Implicit Feedback) ALS -- [MAHOUT-1365](https://issues.apache.org/jira/browse/MAHOUT-1365) -->
<!--* Data frame R-like bindings -- [MAHOUT-1490](https://issues.apache.org/jira/browse/MAHOUT-1490) -->

* *Your issue here!*

<!-- ## Stuff wanted: 
* Data frame R-like bindings (similarly to linalg bindings)
* Stat R-like bindings (perhaps we can just adapt to commons.math stat)
* **BYODMs:** Bring Your Own Distributed Method on SparkBindings! 
* In-core jBlas matrix adapter
* In-core GPU matrix adapters -->



  