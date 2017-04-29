---
layout: default
title: Principal Components Analysis
theme:
    name: retro-mahout
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
