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
title: SVD - Singular Value Decomposition

    
---

{excerpt}Singular Value Decomposition is a form of product decomposition of
a matrix in which a rectangular matrix A is decomposed into a product U s
V' where U and V are orthonormal and s is a diagonal matrix.{excerpt}  The
values of A can be real or complex, but the real case dominates
applications in machine learning.  The most prominent properties of the SVD
are:

  * The decomposition of any real matrix has only real values
  * The SVD is unique except for column permutations of U, s and V
  * If you take only the largest n values of s and set the rest to zero,
you have a least squares approximation of A with rank n.  This allows SVD
to be used very effectively in least squares regression and makes partial
SVD useful.
  * The SVD can be computed accurately for singular or nearly singular
matrices.  For a matrix of rank n, only the first n singular values will be
non-zero.  This allows SVD to be used for solution of singular linear
systems.  The columns of U and V corresponding to zero singular values
define the null space of A.
  * The partial SVD of very large matrices can be computed very quickly
using stochastic decompositions.  See http://arxiv.org/abs/0909.4061v1 for
details.  Gradient descent can also be used to compute partial SVD's and is
very useful where some values of the matrix being decomposed are not known.

In collaborative filtering and text retrieval, it is common to compute the
partial decomposition of the user x item interaction matrix or the document
x term matrix.	This allows the projection of users and items (or documents
and terms) into a common vector space representation that is often referred
to as the latent semantic representation.  This process is sometimes called
Latent Semantic Analysis and has been very effective in the analysis of the
Netflix dataset.

Dimension Reduction in Mahout:
 * https://cwiki.apache.org/MAHOUT/dimensional-reduction.html

 See Also:
 * http://www.kwon3d.com/theory/jkinem/svd.html
 * http://en.wikipedia.org/wiki/Singular_value_decomposition
 * http://en.wikipedia.org/wiki/Latent_semantic_analysis
 * http://en.wikipedia.org/wiki/Netflix_Prize
 *
http://www.amazon.com/Understanding-Complex-Datasets-Decompositions-Knowledge/dp/1584888326
 * http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
 *
http://www.quora.com/What-s-the-best-parallelized-sparse-SVD-code-publicly-available
 * [understanding Mahout Hadoop SVD thread](http://mail-archives.apache.org/mod_mbox/mahout-user/201102.mbox/%3CAANLkTinQ5K4XrM7naBWn8qoBXZGVobBot2RtjZSV4yOd@mail.gmail.com%3E)
