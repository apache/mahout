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
title: Cochrane-Orcutt Procedure

    
---

### About

The [Cochrane Orcutt](https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation) procedure is use in economics to 
adjust a linear model for serial correlation in the error term. 
 
The cooresponding method in R is [`cochrane.orcutt`](https://cran.r-project.org/web/packages/orcutt/orcutt.pdf)
however the implementation differes slightly. 

#### R Prototype:
    library(orcutt)

    df = data.frame(t(data.frame(
        c(20.96,  127.3),
        c(21.40,  130.0),
        c(21.96,  132.7),
        c(21.52,  129.4),
        c(22.39,  135.0),
        c(22.76,  137.1),
        c(23.48,  141.2),
        c(23.66,  142.8),
        c(24.10,  145.5),
        c(24.01,  145.3),
        c(24.54,  148.3),
        c(24.30,  146.4),
        c(25.00,  150.2),
        c(25.64,  153.1),
        c(26.36,  157.3),
        c(26.98,  160.7),
        c(27.52,  164.2),
        c(27.78,  165.6),
        c(28.24,  168.7),
        c(28.78,  171.7))))

    rownames(df) <- NULL
    colnames(df) <- c("y", "x")
    my_lm = lm(y ~ x, data=df)
    coch = cochrane.orcutt(my_lm)

    
The R-implementation is kind of...silly.

The above works- converges at 318 iterations- the transformed DW is   1.72, yet the rho is
 .95882.   After 318 iteartions, this will also report a rho of .95882 (which sugguests SEVERE
 autocorrelation- nothing close to 1.72.

 At anyrate, the real prototype for this is the example from [Applied Linear Statistcal Models
 5th Edition by Kunter, Nachstheim, Neter, and Li](https://www.amazon.com/Applied-Linear-Statistical-Models-Hardcover/dp/B010EWX85C/ref=sr_1_4?ie=UTF8&qid=1493847480&sr=8-4&keywords=applied+linear+statistical+models+5th+edition).  
 
Steps:
1. Normal Regression
2. Estimate <foo>\(\rho\)</foo>
3. Get Estimates of Transformed Equation
4. Step 5: Use Betas from (4) to recalculate model from (1)
5. Step 6: repeat  Step 2 through 5 until a stopping criteria is met. Some models call for convergence-
Kunter et. al reccomend 3 iterations, if you don't achieve desired results, use an alternative method.
 
#### Some additional notes from Applied Linear Statistical Models:
 They also provide some interesting notes on p 494:
 
 1. "Cochrane-Orcutt does not always work properly.  A major reason is that when the error terms
 are positively autocorrelated, the estimate <foo>\(r\)</foo> in (12.22) tends to underestimate the autocorrelation
 parameter <foo>\(\rho\)</foo>.  When this bias is serious, it can significantly reduce the effectiveness of the
 Cochrane-Orcutt approach.
 1. "There exists an approximate relation between the [Durbin Watson test statistic](dw-test.html) <foo>\(\mathbf{D}\)</foo> in (12.14)
 and the estimated autocorrelation paramater <foo>\(r\)</foo> in (12.22):
 <center>\(D ~= 2(1-\rho)\)</center>

 They also note on p492:
 "... If the process does not terminate after one or two iterations, a different procedure
 should be employed."
 This differs from the logic found elsewhere, and the method presented in R where, in the simple
  example in the prototype, the procedure runs for 318 iterations. This is why the default
 maximum iteratoins are 3, and should be left as such.

 Also, the prototype and 'correct answers' are based on the example presented in Kunter et. al on
 p492-4 (including dataset).


### Parameters


<div class="table-striped">
  <table class="table">
    <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default Value</th>
     </tr>
     <tr>
        <td><code>'regressor</code></td>
        <td>Any subclass of <code>org.apache.mahout.math.algorithms.regression.LinearRegressorFitter</code></td>
        <td><code>OrdinaryLeastSquares()</code></td>
     </tr>
     <tr>
        <td><code>'iteratoins</code></td>
        <td>Unlike our friends in R- we stick to the 3 iteration guidance.</td>
        <td>3</td>
     </tr>
     <tr>
        <td><code>'cacheHint</code></td>
        <td>The DRM Cache Hint to use when holding the data in memory between iterations</td>
        <td><code>CacheHint.MEMORY_ONLY</code></td>
     </tr>                 
  </table>
</div>

### Example


    val alsmBlaisdellCo = drmParallelize( dense(
          (20.96,  127.3),
          (21.40,  130.0),
          (21.96,  132.7),
          (21.52,  129.4),
          (22.39,  135.0),
          (22.76,  137.1),
          (23.48,  141.2),
          (23.66,  142.8),
          (24.10,  145.5),
          (24.01,  145.3),
          (24.54,  148.3),
          (24.30,  146.4),
          (25.00,  150.2),
          (25.64,  153.1),
          (26.36,  157.3),
          (26.98,  160.7),
          (27.52,  164.2),
          (27.78,  165.6),
          (28.24,  168.7),
          (28.78,  171.7) ))
    
    val drmY = alsmBlaisdellCo(::, 0 until 1)
    val drmX = alsmBlaisdellCo(::, 1 until 2)

    var coModel = new CochraneOrcutt[Int]().fit(drmX, drmY , ('iterations -> 2))
    
    println(coModel.rhos)
    println(coModel.summary)

