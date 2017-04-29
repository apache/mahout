---
layout: default
title: Locally Weighted Linear Regression
theme:
    name: retro-mahout
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
