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
title:     Stochastic SVD

   
---

# Stochastic Singular Value Decomposition #

Stochastic SVD method in Mahout produces reduced rank Singular Value Decomposition output in its 
strict mathematical definition: ` \(\mathbf{A\approx U}\boldsymbol{\Sigma}\mathbf{V}^{\top}\)`.

##The benefits over other methods are:

 - reduced flops required compared to Krylov subspace methods

 - In map-reduce world, a fixed number of MR iterations required regardless of rank requested

 - Tweak precision/speed balance with options.

 - A is a Distributed Row Matrix where rows may be identified by any Writable (such as a document path). As such, it would work directly on the output of seq2sparse.

 - As of 0.7 trunk, includes PCA and dimensionality reduction workflow (EXPERIMENTAL! Feedback on performance/other PCA related issues/ blogs is greatly appreciated.)

### Map-Reduce characteristics: 
SSVD uses at most 3 MR sequential steps (map-only + map-reduce + 2 optional parallel map-reduce jobs) to produce reduced rank approximation of U, V and S matrices. Additionally, two more map-reduce steps are added for each power iteration step if requested.

##Potential drawbacks:

potentially less precise (but adding even one power iteration seems to fix that quite a bit).

##Documentation

[Overview and Usage][3]

Note: Please use 0.6 or later! for PCA workflow, please use 0.7 or later.

##Publications

[Nathan Halko's dissertation][1] "Randomized methods for computing low-rank
approximations of matrices" contains comprehensive definition of parallelization strategy taken in Mahout SSVD implementation and also some precision/scalability benchmarks, esp. w.r.t. Mahout Lanczos implementation on a typical corpus data set.

[Halko, Martinsson, Tropp] paper discusses family of random projection-based algorithms and contains theoretical error estimates.

**R simulation**

[Non-parallel SSVD simulation in R][2] with power iterations and PCA options. Note that this implementation is not most optimal for sequential flow solver, but it is for demonstration purposes only.

However, try this R code to simulate a meaningful input:



**tests.R**



    n<-1000
    m<-2000
    k<-10
     
    qi<-1
     
    #simulated input
    svalsim<-diag(k:1)
     
    usim<- qr.Q(qr(matrix(rnorm(m*k, mean=3), nrow=m,ncol=k)))
    vsim<- qr.Q(qr( matrix(rnorm(n*k,mean=5), nrow=n,ncol=k)))
     
     
    x<- usim %*% svalsim %*% t(vsim)


and try to compare ssvd.svd(x) and stock svd(x) performance for the same rank k, notice the difference in the running time. Also play with power iterations (qIter) and compare accuracies of standard svd and SSVD.

Note: numerical stability of R algorithms may differ from that of Mahout's distributed version. We haven't studied accuracy of the R simulation. For study of accuracy of Mahout's version, please refer to Nathan's dissertation as referenced above.


  [1]: http://amath.colorado.edu/faculty/martinss/Pubs/2012_halko_dissertation.pdf
  [2]: ssvd.page/ssvd.R
  [3]: ssvd.page/SSVD-CLI.pdf


#### Modified SSVD Algorithm.

Given an `\(m\times n\)`
matrix `\(\mathbf{A}\)`, a target rank `\(k\in\mathbb{N}_{1}\)`
, an oversampling parameter `\(p\in\mathbb{N}_{1}\)`, 
and the number of additional power iterations `\(q\in\mathbb{N}_{0}\)`, 
this procedure computes an `\(m\times\left(k+p\right)\)`
SVD `\(\mathbf{A\approx U}\boldsymbol{\Sigma}\mathbf{V}^{\top}\)`:

  1. Create seed for random `\(n\times\left(k+p\right)\)`
  matrix `\(\boldsymbol{\Omega}\)`. The seed defines matrix `\(\mathbf{\Omega}\)`
  using Gaussian unit vectors per one of suggestions in [Halko, Martinsson, Tropp].

  2. `\(\mathbf{Y=A\boldsymbol{\Omega}},\,\mathbf{Y}\in\mathbb{R}^{m\times\left(k+p\right)}\)`
 

  3. Column-orthonormalize `\(\mathbf{Y}\rightarrow\mathbf{Q}\)`
  by computing thin decomposition `\(\mathbf{Y}=\mathbf{Q}\mathbf{R}\)`.
  Also, `\(\mathbf{Q}\in\mathbb{R}^{m\times\left(k+p\right)},\,\mathbf{R}\in\mathbb{R}^{\left(k+p\right)\times\left(k+p\right)}\)`.
  I denote this as `\(\mathbf{Q}=\mbox{qr}\left(\mathbf{Y}\right).\mathbf{Q}\)`
 

  4. `\(\mathbf{B}_{0}=\mathbf{Q}^{\top}\mathbf{A}:\,\,\mathbf{B}\in\mathbb{R}^{\left(k+p\right)\times n}\)`.
 
  5. If `\(q>0\)`
  repeat: for `\(i=1..q\)`: 
  `\(\mathbf{B}_{i}^{\top}=\mathbf{A}^{\top}\mbox{qr}\left(\mathbf{A}\mathbf{B}_{i-1}^{\top}\right).\mathbf{Q}\)`
  (power iterations step).

  6. Compute Eigensolution of a small Hermitian `\(\mathbf{B}_{q}\mathbf{B}_{q}^{\top}=\mathbf{\hat{U}}\boldsymbol{\Lambda}\mathbf{\hat{U}}^{\top}\)`,
  `\(\mathbf{B}_{q}\mathbf{B}_{q}^{\top}\in\mathbb{R}^{\left(k+p\right)\times\left(k+p\right)}\)`.
 

  7. Singular values `\(\mathbf{\boldsymbol{\Sigma}}=\boldsymbol{\Lambda}^{0.5}\)`,
  or, in other words, `\(s_{i}=\sqrt{\sigma_{i}}\)`.
 

  8. If needed, compute `\(\mathbf{U}=\mathbf{Q}\hat{\mathbf{U}}\)`.
 

  9. If needed, compute `\(\mathbf{V}=\mathbf{B}_{q}^{\top}\hat{\mathbf{U}}\boldsymbol{\Sigma}^{-1}\)`.
Another way is `\(\mathbf{V}=\mathbf{A}^{\top}\mathbf{U}\boldsymbol{\Sigma}^{-1}\)`.

[Halko, Martinsson, Tropp]: http://arxiv.org/abs/0909.4061
 
