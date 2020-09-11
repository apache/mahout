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
title: (Deprecated)  Support Vector Machines

    
---

<a name="SupportVectorMachines-SupportVectorMachines"></a>
# Support Vector Machines

As with Naive Bayes, Support Vector Machines (or SVMs in short) can be used
to solve the task of assigning objects to classes. However, the way this
task is solved is completely different to the setting in Naive Bayes.

Each object is considered to be a point in _n_ dimensional feature space,
_n_ being the number of features used to describe the objects numerically.
In addition each object is assigned a binary label, let us assume the
labels are "positive" and "negative". During learning, the algorithm tries
to find a hyperplane in that space, that perfectly separates positive from
negative objects.
It is trivial to think of settings where this might very well be
impossible. To remedy this situation, objects can be assigned so called
slack terms, that punish mistakes made during learning appropriately. That
way, the algorithm is forced to find the hyperplane that causes the least
number of mistakes.

Another way to overcome the problem of there being no linear hyperplane to
separate positive from negative objects is to simply project each feature
vector into an higher dimensional feature space and search for a linear
separating hyperplane in that new space. Usually the main problem with
learning in high dimensional feature spaces is the so called curse of
dimensionality. That is, there are fewer learning examples available than
free parameters to tune. In the case of SVMs this problem is less
detrimental, as SVMs impose additional structural constraints on their
solutions. Each separating hyperplane needs to have a maximal margin to all
training examples. In addition, that way, the solution may be based on the
information encoded in only very few examples.

<a name="SupportVectorMachines-Strategyforparallelization"></a>
## Strategy for parallelization

<a name="SupportVectorMachines-Designofpackages"></a>
## Design of packages
