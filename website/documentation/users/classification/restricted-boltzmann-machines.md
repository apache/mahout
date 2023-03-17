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
title: Restricted Boltzmann Machines

    
---

NOTE: This implementation is a Work-In-Progress, at least till September,
2010. 

The JIRA issue is [here](https://issues.apache.org/jira/browse/MAHOUT-375)
. 

<a name="RestrictedBoltzmannMachines-BoltzmannMachines"></a>
### Boltzmann Machines
Boltzmann Machines are a type of stochastic neural networks that closely
resemble physical processes. They define a network of units with an overall
energy that is evolved over a period of time, until it reaches thermal
equilibrium. 

However, the convergence speed of Boltzmann machines that have
unconstrained connectivity is low.

<a name="RestrictedBoltzmannMachines-RestrictedBoltzmannMachines"></a>
### Restricted Boltzmann Machines
Restricted Boltzmann Machines are a variant, that are 'restricted' in the
sense that connections between hidden units of a single layer are _not_
allowed. In addition, stacking multiple RBM's is also feasible, with the
activities of the hidden units forming the base for a higher-level RBM. The
combination of these two features renders RBM's highly usable for
parallelization. 

In the Netflix Prize, RBM's offered distinctly orthogonal predictions to
SVD and k-NN approaches, and contributed immensely to the final solution.

<a name="RestrictedBoltzmannMachines-RBM'sinApacheMahout"></a>
### RBM's in Apache Mahout
An implementation of Restricted Boltzmann Machines is being developed for
Apache Mahout as a Google Summer of Code 2010 project. A recommender
interface will also be provided. The key aims of the implementation are:
1. Accurate - should replicate known results, including those of the Netflix
Prize
1. Fast - The implementation uses Map-Reduce, hence, it should be fast
1. Scale - Should scale to large datasets, with a design whose critical
parts don't need a dependency between the amount of memory on your cluster
systems and the size of your dataset

You can view the patch as it develops [here](http://github.com/sisirkoppaka/mahout-rbm/compare/trunk...rbm)
.
