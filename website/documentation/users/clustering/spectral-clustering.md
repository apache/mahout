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
title: Spectral Clustering

   
---

# Spectral Clustering Overview

Spectral clustering, as its name implies, makes use of the spectrum (or eigenvalues) of the similarity matrix of the data. It examines the _connectedness_ of the data, whereas other clustering algorithms such as k-means use the _compactness_ to assign clusters. Consequently, in situations where k-means performs well, spectral clustering will also perform well. Additionally, there are situations in which k-means will underperform (e.g. concentric circles), but spectral clustering will be able to segment the underlying clusters. Spectral clustering is also very useful for image segmentation.

At its simplest, spectral clustering relies on the following four steps:

 1. Computing a similarity (or _affinity_) matrix `\(\mathbf{A}\)` from the data. This involves determining a pairwise distance function `\(f\)` that takes a pair of data points and returns a scalar.

 2. Computing a graph Laplacian `\(\mathbf{L}\)` from the affinity matrix. There are several types of graph Laplacians; which is used will often depends on the situation.

 3. Computing the eigenvectors and eigenvalues of `\(\mathbf{L}\)`. The degree of this decomposition is often modulated by `\(k\)`, or the number of clusters. Put another way, `\(k\)` eigenvectors and eigenvalues are computed.

 4. The `\(k\)` eigenvectors are used as "proxy" data for the original dataset, and fed into k-means clustering. The resulting cluster assignments are transparently passed back to the original data.

For more theoretical background on spectral clustering, such as how affinity matrices are computed, the different types of graph Laplacians, and whether the top or bottom eigenvectors and eigenvalues are computed, please read [Ulrike von Luxburg's article in _Statistics and Computing_ from December 2007](http://link.springer.com/article/10.1007/s11222-007-9033-z). It provides an excellent description of the linear algebra operations behind spectral clustering, and imbues a thorough understanding of the types of situations in which it can be used.

# Mahout Spectral Clustering

As of Mahout 0.3, spectral clustering has been implemented to take advantage of the MapReduce framework. It uses [SSVD](http://mahout.apache.org/users/dim-reduction/ssvd.html) for dimensionality reduction of the input data set, and [k-means](http://mahout.apache.org/users/clustering/k-means-clustering.html) to perform the final clustering.

**([MAHOUT-1538](https://issues.apache.org/jira/browse/MAHOUT-1538) will port the existing Hadoop MapReduce implementation to Mahout DSL, allowing for one of several distinct distributed back-ends to conduct the computation)**

## Input

The input format for the algorithm currently takes the form of a Hadoop-backed affinity matrix in the form of text files. Each line of the text file specifies a single element of the affinity matrix: the row index `\(i\)`, the column index `\(j\)`, and the value:

`i, j, value`

The affinity matrix is symmetric, and any unspecified `\(i, j\)` pairs are assumed to be 0 for sparsity. The row and column indices are 0-indexed. Thus, only the non-zero entries of either the upper or lower triangular need be specified.

The matrix elements specified in the text files are collected into a Mahout `DistributedRowMatrix`.

**([MAHOUT-1539](https://issues.apache.org/jira/browse/MAHOUT-1539) will allow for the creation of the affinity matrix to occur as part of the core spectral clustering algorithm, as opposed to the current requirement that the user create this matrix themselves and provide it, rather than the original data, to the algorithm)**

## Running spectral clustering

**([MAHOUT-1540](https://issues.apache.org/jira/browse/MAHOUT-1540) will provide a running example of this algorithm and this section will be updated to show how to run the example and what the expected output should be; until then, this section provides a how-to for simply running the algorithm on arbitrary input)**

Spectral clustering can be invoked with the following arguments.

    bin/mahout spectralkmeans \
        -i <affinity matrix directory> \
        -o <output working directory> \
        -d <number of data points> \
        -k <number of clusters AND number of top eigenvectors to use> \
        -x <maximum number of k-means iterations>

The affinity matrix can be contained in a single text file (using the aforementioned one-line-per-entry format) or span many text files [per (MAHOUT-978](https://issues.apache.org/jira/browse/MAHOUT-978), do not prefix text files with a leading underscore '_' or period '.'). The `-d` flag is required for the algorithm to know the dimensions of the affinity matrix. `-k` is the number of top eigenvectors from the normalized graph Laplacian in the SSVD step, and also the number of clusters given to k-means after the SSVD step.

## Example

To provide a simple example, take the following affinity matrix, contained in a text file called `affinity.txt`:

    0, 0, 0
    0, 1, 0.8
    0, 2, 0.5
    1, 0, 0.8
    1, 1, 0
    1, 2, 0.9
    2, 0, 0.5
    2, 1, 0.9
    2, 2, 0

With this 3-by-3 matrix, `-d` would be `3`. Furthermore, since all affinity matrices are assumed to be symmetric, the entries specifying both `1, 2, 0.9` and `2, 1, 0.9` are redundant; only one of these is needed. Additionally, any entries that are 0, such as those along the diagonal, also need not be specified at all. They are provided here for completeness.

In general, larger values indicate a stronger "connectedness", whereas smaller values indicate a weaker connectedness. This will vary somewhat depending on the distance function used, though a common one is the [RBF kernel](http://en.wikipedia.org/wiki/RBF_kernel) (used in the above example) which returns values in the range [0, 1], where 0 indicates completely disconnected (or completely dissimilar) and 1 is fully connected (or identical).

The call signature with this matrix could be as follows:

    bin/mahout spectralkmeans \
        -i s3://mahout-example/input/ \
        -o s3://mahout-example/output/ \
        -d 3 \
        -k 2 \
        -x 10

There are many other optional arguments, in particular for tweaking the SSVD process (block size, number of power iterations, etc) and the k-means clustering step (distance measure, convergence delta, etc).