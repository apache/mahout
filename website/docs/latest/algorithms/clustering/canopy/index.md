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
title: Canopy Clustering

   
---

### About

[Canopy Clustering](http://www.kamalnigam.com/papers/canopy-kdd00.pdf)
 is a very simple, fast and surprisingly accurate method for grouping
objects into clusters. All objects are represented as a point in a
multidimensional feature space. The algorithm uses a fast approximate
distance metric and two distance thresholds T1 > T2 for processing. The
basic algorithm is to begin with a set of points and remove one at random.
Create a Canopy containing this point and iterate through the remainder of
the point set. At each point, if its distance from the first point is < T1,
then add the point to the cluster. If, in addition, the distance is < T2,
then remove the point from the set. This way points that are very close to
the original will avoid all further processing. The algorithm loops until
the initial set is empty, accumulating a set of Canopies, each containing
one or more points. A given point may occur in more than one Canopy.

Canopy Clustering is often used as an initial step in more rigorous
clustering techniques, such as [K-Means Clustering](k-means-clustering.html)
. By starting with an initial clustering the number of more expensive
distance measurements can be significantly reduced by ignoring points
outside of the initial canopies.

#### Strategy for parallelization

Looking at the sample Hadoop implementation in [http://code.google.com/p/canopy-clustering/](http://code.google.com/p/canopy-clustering/)
 the processing is done in 3 steps:
1. The data is massaged into suitable input format
1. Each mapper performs canopy clustering on the points in its input set and
outputs its canopies' centers
1. The reducer clusters the canopy centers to produce the final canopy
centers

The points are then clustered into these final canopies when the `model.cluster(inputDRM)` is called.

Some ideas can be found in [Cluster computing and MapReduce](https://www.youtube.com/watch?v=yjPBkvYh-ss&list=PLEFAB97242917704A)
 lecture video series \[by Google(r)\]; Canopy Clustering is discussed in [lecture #4](https://www.youtube.com/watch?v=1ZDybXl212Q)
. Finally here is the [Wikipedia page](http://en.wikipedia.org/wiki/Canopy_clustering_algorithm)
.

#### Illustrations

The following images illustrate Canopy clustering applied to a set of
randomly-generated 2-d data points. The points are generated using a normal
distribution centered at a mean location and with a constant standard
deviation. See the README file in the [/examples/src/main/java/org/apache/mahout/clustering/display/README.txt](https://github.com/apache/mahout/blob/master/examples/src/main/java/org/apache/mahout/clustering/display/README.txt)
 for details on running similar examples.

The points are generated as follows:

* 500 samples m=\[1.0, 1.0\](1.0,-1.0\.html)
 sd=3.0
* 300 samples m=\[1.0, 0.0\](1.0,-0.0\.html)
 sd=0.5
* 300 samples m=\[0.0, 2.0\](0.0,-2.0\.html)
 sd=0.1

In the first image, the points are plotted and the 3-sigma boundaries of
their generator are superimposed. 

![sample data](SampleData.png)

In the second image, the resulting canopies are shown superimposed upon the
sample data. Each canopy is represented by two circles, with radius T1 and
radius T2.

![canopy](Canopy.png)

The third image uses the same values of T1 and T2 but only superimposes
canopies covering more than 10% of the population. This is a bit better
representation of the data but it still has lots of room for improvement.
The advantage of Canopy clustering is that it is single-pass and fast
enough to iterate runs using different T1, T2 parameters and display
thresholds.

![canopy](Canopy10.png)

### Parameters

<div class="table-striped">
  <table class="table">
    <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default Value</th>
    </tr>
    <tr>
        <td><code>'distanceMeasure</code></td>
        <td>The metric used for calculating distance, see <a href="../distance-metrics.html">Distance Metrics</a></td>
        <td><code>'Cosine</code></td>
    </tr>
    <tr>
        <td><code>'t1</code></td>
        <td>The "loose" distance in the mapping phase</code></td>
        <td>0.5</td>
    </tr>
    <tr>
        <td><code>'t2</code></td>
        <td>The "tight" distance in the mapping phase</code></td>
        <td>0.1</td>
    </tr>
    <tr>
        <td><code>'t3</code></td>
        <td>The "loose" distance in the reducing phase</code></td>
        <td><code>'t1</code></td>
    </tr>
    <tr>
        <td><code>'t4</code></td>
        <td>The "tight" distance in the reducing phase</code></td>
        <td><code>'t2</code></td>
    </tr>
  </table>
</div>

### Example

    val drmA = drmParallelize(dense((1.0, 1.2, 1.3, 1.4), (1.1, 1.5, 2.5, 1.0), (6.0, 5.2, -5.2, 5.3), (7.0,6.0, 5.0, 5.0), (10.0, 1.0, 20.0, -10.0)))
    
    import org.apache.mahout.math.algorithms.clustering.CanopyClustering
    
    val model = new CanopyClustering().fit(drmA, 't1 -> 6.5, 't2 -> 5.5, 'distanceMeasure -> 'Chebyshev)
    model.cluster(drmA).collect
