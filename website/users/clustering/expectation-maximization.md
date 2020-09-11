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
title: Expectation Maximization

   
---
<a name="ExpectationMaximization-ExpectationMaximization"></a>
# Expectation Maximization

The principle of EM can be applied to several learning settings, but is
most commonly associated with clustering. The main principle of the
algorithm is comparable to k-Means. Yet in contrast to hard cluster
assignments, each object is given some probability to belong to a cluster.
Accordingly cluster centers are recomputed based on the average of all
objects weighted by their probability of belonging to the cluster at hand.

<a name="ExpectationMaximization-Canopy-modifiedEM"></a>
## Canopy-modified EM

One can also use the canopies idea to speed up prototypebased clustering
methods like K-means and Expectation-Maximization (EM). In general, neither
K-means nor EMspecify how many clusters to use. The canopies technique does
not help this choice.

Prototypes (our estimates of the cluster centroids) are associated with the
canopies that contain them, and the prototypes are only influenced by data
that are inside their associated canopies. After creating the canopies, we
decide how many prototypes will be created for each canopy. This could be
done, for example, using the number of data points in a canopy and AIC or
BIC where points that occur in more than one canopy are counted
fractionally. Then we place prototypesinto each canopy. This initial
placement can be random, as long as it is within the canopy in question, as
determined by the inexpensive distance metric.

Then, instead of calculating the distance from each prototype to every
point (as is traditional, a O(nk) operation), theE-step instead calculates
the distance from each prototype to a much smaller number of points. For
each prototype, we find the canopies that contain it (using the cheap
distance metric), and only calculate distances (using the expensive
distance metric) from that prototype to points within those canopies.

Note that by this procedure prototypes may move across canopy boundaries
when canopies overlap. Prototypes may move to cover the data in the
overlapping region, and then move entirely into another canopy in order to
cover data there.

The canopy-modified EM algorithm behaves very similarly to traditional EM,
with the slight difference that points outside the canopy have no influence
on points in the canopy, rather than a minute influence. If the canopy
property holds, and points in the same cluster fall in the same canopy,
then the canopy-modified EM will almost always converge to the same maximum
in likelihood as the traditional EM. In fact, the difference in each
iterative step (apart from the enormous computational savings of computing
fewer terms) will be negligible since points outside the canopy will have
exponentially small influence.

<a name="ExpectationMaximization-StrategyforParallelization"></a>
## Strategy for Parallelization

<a name="ExpectationMaximization-Map/ReduceImplementation"></a>
## Map/Reduce Implementation

