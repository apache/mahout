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
title: Algorithms

    
---


---
*Mahout 0.12.0 Features by Engine*   
---

| | **Single Machine** | [**MapReduce**](http://hadoop.apache.org/)| [**Spark**](https://spark.apache.org/) | [**H2O**](http://0xdata.com/) | [**Flink**](https://flink.apache.org/) |
---------------------------------------------|:----------------:|:-----------:|:------:|:---:|:----:|
**Mahout Math-Scala Core Library and Scala DSL**| 
|   [Mahout Distributed BLAS. Distributed Row Matrix API with R and Matlab like operators. Distributed ALS, SPCA, SSVD, thin-QR. Similarity Analysis](http://mahout.apache.org/users/sparkbindings/home.html).    | |  | [x](https://mahout.apache.org/users/sparkbindings/ScalaSparkBindings.pdf) | [x](https://github.com/apache/mahout/tree/master/h2o) |[x](https://github.com/apache/mahout/tree/flink-binding/flink)
||
**Mahout Interactive Shell**| 
|   [Interactive REPL shell for Spark optimized Mahout DSL](http://mahout.apache.org/users/sparkbindings/play-with-shell.html) | | | x |
||
**Collaborative Filtering** *with CLI drivers*|
    User-Based Collaborative Filtering           | *deprecated* | *deprecated*|[x](https://mahout.apache.org/users/algorithms/intro-cooccurrence-spark.html)
    Item-Based Collaborative Filtering           | x | [x](https://mahout.apache.org/users/recommender/intro-itembased-hadoop.html) | [x](https://mahout.apache.org/users/algorithms/intro-cooccurrence-spark.html) |
    Matrix Factorization with ALS | x | [x](https://mahout.apache.org/users/recommender/intro-als-hadoop.html) |  |
    Matrix Factorization with ALS on Implicit Feedback | x | [x](https://mahout.apache.org/users/recommender/intro-als-hadoop.html) |  |
    Weighted Matrix Factorization, SVD++  | x | | 
||
**Classification** *with CLI drivers*| | |
    Logistic Regression - trained via SGD   | [*deprecated*](http://mahout.apache.org/users/classification/logistic-regression.html) |
    Naive Bayes / Complementary Naive Bayes  | | [*deprecated*](https://mahout.apache.org/users/classification/bayesian.html) | [x](https://mahout.apache.org/users/algorithms/spark-naive-bayes.html) | 
    Hidden Markov Models   | [*deprecated*](https://mahout.apache.org/users/classification/hidden-markov-models.html) |
||
**Clustering** *with CLI drivers*||
    Canopy Clustering  | [*deprecated*](https://mahout.apache.org/users/clustering/canopy-clustering.html) | [*deprecated*](https://mahout.apache.org/users/clustering/canopy-clustering.html)| 
    k-Means Clustering   | [*deprecated*](https://mahout.apache.org/users/clustering/k-means-clustering.html) | [*deprecated*](https://mahout.apache.org/users/clustering/k-means-clustering.html) |  
    Fuzzy k-Means   | [*deprecated*](https://mahout.apache.org/users/clustering/fuzzy-k-means.html) | [*deprecated*](https://mahout.apache.org/users/clustering/fuzzy-k-means.html)|  
    Streaming k-Means   | [*deprecated*](https://mahout.apache.org/users/clustering/streaming-k-means.html) | [*deprecated*](https://mahout.apache.org/users/clustering/streaming-k-means.html) |  
    Spectral Clustering   |  | [*deprecated*](https://mahout.apache.org/users/clustering/spectral-clustering.html) |  
||
**Dimensionality Reduction** *note: most scala-based dimensionality reduction algorithms are available through the [Mahout Math-Scala Core Library for all engines](https://mahout.apache.org/users/sparkbindings/home.html)*||
    Singular Value Decomposition | *deprecated* | *deprecated* | [x](http://mahout.apache.org/users/sparkbindings/home.html) |[x](http://mahout.apache.org/users/environment/h2o-internals.html) |   [x](http://mahout.apache.org/users/flinkbindings/flink-internals.html)
    Lanczos Algorithm  | *deprecated* | *deprecated* | 
    Stochastic SVD  | [*deprecated*](https://mahout.apache.org/users/dim-reduction/ssvd.html) | [*deprecated*](https://mahout.apache.org/users/dim-reduction/ssvd.html) | [x](http://mahout.apache.org/users/algorithms/d-ssvd.html) | [x](http://mahout.apache.org/users/algorithms/d-ssvd.html)|    [x](http://mahout.apache.org/users/algorithms/d-ssvd.html)
    PCA (via Stochastic SVD) | *deprecated* | *deprecated* | [x](http://mahout.apache.org/users/sparkbindings/home.html)  |[x](http://mahout.apache.org/users/environment/h2o-internals.html) |   [x](http://mahout.apache.org/users/flinkbindings/flink-internals.html)
    QR Decomposition         | *deprecated* | *deprecated* | [x](http://mahout.apache.org/users/algorithms/d-qr.html) |[x](http://mahout.apache.org/users/algorithms/d-qr.html) |   [x](http://mahout.apache.org/users/algorithms/d-qr.html)
||
**Topic Models**||
    Latent Dirichlet Allocation  | *deprecated* | *deprecated* |
||
**Miscellaneous**||
    RowSimilarityJob   |  | *deprecated* | [x](https://github.com/apache/mahout/blob/master/spark/src/test/scala/org/apache/mahout/drivers/RowSimilarityDriverSuite.scala) |
    Collocations  |  | [*deprecated*](https://mahout.apache.org/users/basics/collocations.html) |  
    Sparse TF-IDF Vectors from Text |  | [*deprecated*](https://mahout.apache.org/users/basics/creating-vectors-from-text.html) |
    XML Parsing|  | [*deprecated*](https://issues.apache.org/jira/browse/MAHOUT-1479?jql=text%20~%20%22wikipedia%20mahout%22) |
    Email Archive Parsing |  | [*deprecated*](https://github.com/apache/mahout/tree/master/integration/src/main/java/org/apache/mahout/text) | 
    Evolutionary Processes | [x](https://github.com/apache/mahout/tree/master/mr/src/main/java/org/apache/mahout/ep) |
    

