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
title: Ordinary Least Squares Regression

    
---

### About

The `OrinaryLeastSquares` regressor in Mahout implements a _closed-form_ solution to [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares). 
This is in stark contrast to many "big data machine learning" frameworks which implement a _stochastic_ approach. From the users perspecive this difference can be reduced to:

- **_Stochastic_**- A series of guesses at a line line of best fit. 
- **_Closed Form_**- A mathimatical approach has been explored, the properties of the parameters are well understood, and problems which arise (and the remedial measures), exist.  This is usually the preferred choice of mathematicians/statisticians, but computational limititaions have forced us to resort to SGD.

### Parameters

<div class="table-striped">
  <table class="table">
    <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default Value</th>
     </tr>
     <tr>
        <td><code>'calcCommonStatistics</code></td>
        <td>Calculate commons statistics such as Coeefficient of Determination and Mean Square Error</td>
        <td><code>true</code></td>
     </tr>
     <tr>
        <td><code>'calcStandardErrors</code></td>
        <td>Calculate the standard errors (and subsequent "t-scores" and "p-values") of the \(\boldsymbol{\beta}\) estimates</td>
        <td><code>true</code></td>
     </tr>
     <tr>
        <td><code>'addIntercept</code></td>
        <td>Add an intercept to \(\mathbf{X}\)</td>
        <td><code>true</code></td>
     </tr>                 
  </table>
</div>

### Example

In this example we disable the "calculate common statistics" parameters, so our summary will NOT contain the coefficient of determination (R-squared) or Mean Square Error
```scala
import org.apache.mahout.math.algorithms.regression.OrdinaryLeastSquares

val drmData = drmParallelize(dense(
      (2, 2, 10.5, 10, 29.509541),  // Apple Cinnamon Cheerios
      (1, 2, 12,   12, 18.042851),  // Cap'n'Crunch
      (1, 1, 12,   13, 22.736446),  // Cocoa Puffs
      (2, 1, 11,   13, 32.207582),  // Froot Loops
      (1, 2, 12,   11, 21.871292),  // Honey Graham Ohs
      (2, 1, 16,   8,  36.187559),  // Wheaties Honey Gold
      (6, 2, 17,   1,  50.764999),  // Cheerios
      (3, 2, 13,   7,  40.400208),  // Clusters
      (3, 3, 13,   4,  45.811716)), numPartitions = 2)


val drmX = drmData(::, 0 until 4)
val drmY = drmData(::, 4 until 5)

val model = new OrdinaryLeastSquares[Int]().fit(drmX, drmY, 'calcCommonStatistics → false)
println(model.summary)
```
