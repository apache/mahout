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
title: StandardScaler

    
---

### About

`StandardScaler` centers the values of each column to their mean, and scales them to unit variance. 

#### Relation to the `scale` function in R-base
The `StandardScaler` is the equivelent of the R-base function [`scale`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/scale.html) with
one noteable tweek. R's `scale` function (indeed all of R) calculates standard deviation with 1 degree of freedom, Mahout 
(like many other statistical packages aimed at larger data sets) does not make this adjustment.  In larger datasets the difference
is trivial, however when testing the function on smaller datasets the practicioner may be confused by the discrepency. 

To verify this function against R on an arbitrary matrix, use the following form in R to "undo" the degrees of freedom correction.
```R
N <- nrow(x)
scale(x, scale= apply(x, 2, sd) * sqrt(N-1/N))
```

### Parameters

`StandardScaler` takes no parameters at this time.

### Example


```scala
import org.apache.mahout.math.algorithms.preprocessing.StandardScaler

val A = drmParallelize(dense(
      (1, 1, 5),
      (2, 5, -15),
      (3, 9, -2)), numPartitions = 2)

val scaler: StandardScalerModel = new StandardScaler().fit(A)

val scaledA = scaler.transform(A)
```


