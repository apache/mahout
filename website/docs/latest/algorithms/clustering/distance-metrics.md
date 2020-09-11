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
title: Distance Metrics

   
---

### Distance Metrics Supported By Mahout

<div class="table-striped">
  <table class="table">
    <tr>
        <th>Name</th>
        <th>Object</th>
        <th>Symbol</th>
    </tr>
    <tr>
        <td><a href="https://en.wikipedia.org/wiki/Chebyshev_distance">Chebyshev Distance</a></td>
        <td><code>org.apache.mahout.math.algorithms.common.distance.Chebyshev</code></td>
        <td><code>'Chebyshev</code></td>
    </tr>
    <tr>
        <td><a href="https://en.wikipedia.org/wiki/Cosine_similarity">Cosine Similarity</a></td>
        <td><code>org.apache.mahout.math.algorithms.common.distance.Cosine</code></td>
        <td><code>'Cosine</code></td>
    </tr>
    </table>
</div>


<!--
A beginner JIRA to port the rest of these
[Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance)

[Mahalanobis](https://en.wikipedia.org/wiki/Mahalanobis_distance)

[Manhattan](https://en.wiktionary.org/wiki/Manhattan_distance)

[Minkowski](https://en.wikipedia.org/wiki/Minkowski_distance)

[Squared Euclidian](https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance)

[Tanimoto](https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance)

Weighted Euclidean

Weighted Manhattan-->

### Using Distance Metrics

In Mahout one can access the distant metrics directly to measure the distance between two arbitrary vectors, or 
can specify which distance metric to use as part of an algorithm. In the latter case the distance metric is called 
by `Symbol`, we never pass Distance metrics directly to an algorithm.  This design choice, in part has to do with 
serialization of object and keeping the engine bindings as simple as possible.  Behind the scenes, the only thing 
that is serialized and sent to the workers is a number which specifies what distant metric to use- this is much more
abstract and easier to maintain on the back end than making sure each function can be serialized by any arbitrary engine.
We feel from the user perspective, it may seem quirky but causes no decrease in usability.  If a user wishes to use a 
custom distance metric- simply add it to [math-scala/src/main/org/apache/mahout/math/common/DistanceMetrics.scala](https://github.com/apache/mahout/blob/master/math-scala/src/main/scala/org/apache/mahout/math/algorithms/common/DistanceMetrics.scala)
and recompile. 

### Examples

**Meausring the distance between two vectors**

    import org.apache.mahout.math.algorithms.common.distance._
    
    val v1 = dvec(1.0, 1.5, -1.2, 3.5)
    val v2 = dvec(0.1, -1.4, 10.5, 3.2)
    
    Cosine.distance(v1, v2)

**Using distance in clustering**

    import org.apache.mahout.math.algorithms.clustering.CanopyClustering

    val drmA = drmParallelize(dense((1.0, 1.2, 1.3, 1.4), 
                                    (1.1, 1.5, 2.5, 1.0), 
                                    (6.0, 5.2, -5.2, 5.3), 
                                    (7.0,6.0, 5.0, 5.0), 
                                    (10.0, 1.0, 20.0, -10.0)))
                                    
    val model = new CanopyClustering().fit(drmA, 'distanceMeasure -> 'Cosine)