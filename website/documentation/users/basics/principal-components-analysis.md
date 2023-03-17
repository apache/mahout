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
title: Principal Components Analysis

    
---

<a name="PrincipalComponentsAnalysis-PrincipalComponentsAnalysis"></a>
# Principal Components Analysis

PCA is used to reduce high dimensional data set to lower dimensions. PCA
can be used to identify patterns in data, express the data in a lower
dimensional space. That way, similarities and differences can be
highlighted. It is mostly used in face recognition and image compression.
There are several flaws one has to be aware of when working with PCA:

* Linearity assumption - data is assumed to be linear combinations of some
basis. There exist non-linear methods such as kernel PCA that alleviate
that problem.
* Principal components are assumed to be orthogonal. ICA tries to cope with
this limitation.
* Mean and covariance are assumed to be statistically important.
* Large variances are assumed to have important dynamics.

<a name="PrincipalComponentsAnalysis-Parallelizationstrategy"></a>
## Parallelization strategy

<a name="PrincipalComponentsAnalysis-Designofpackages"></a>
## Design of packages
