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
title: (Deprecated)  Locally Weighted Linear Regression

    
---

<a name="LocallyWeightedLinearRegression-LocallyWeightedLinearRegression"></a>
# Locally Weighted Linear Regression

Model-based methods, such as SVM, Naive Bayes and the mixture of Gaussians,
use the data to build a parameterized model. After training, the model is
used for predictions and the data are generally discarded. In contrast,
"memory-based" methods are non-parametric approaches that explicitly retain
the training data, and use it each time a prediction needs to be made.
Locally weighted regression (LWR) is a memory-based method that performs a
regression around a point of interest using only training data that are
"local" to that point. Source:
http://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/cohn96a-html/node7.html

<a name="LocallyWeightedLinearRegression-Strategyforparallelregression"></a>
## Strategy for parallel regression

<a name="LocallyWeightedLinearRegression-Designofpackages"></a>
## Design of packages
