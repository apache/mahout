---
layout: doc-page
title: Ridge Regression
redirect_from:
   - /docs/latest/algorithms/regression/ridge
   - /docs/latest/algorithms/regression/ridge.html
---

### About

The `Ridge` regressor in Mahout implements a _closed-form_ solution to Ridge Regression (https://en.wikipedia.org/wiki/Tikhonov_regularization).
Based on the linear regressor, Ridge regression adds, from a bayesian perspective, a prior normal distribution to the _beta_ coefficients matrix,
centered with a standard deviation given by the _lambda_ parameter. The higher the lambda value, the more _spread_ the beta values should be.
From a linear algebra perspective, the addition of a diagonal matrix determined by the lambda hyperparameter, breaks matrix collinearity, thus
making the design matrix product invertible. Finally, from an optimization perspective, a higher value of _lambda_ penalizes higher values of beta
coefficients since it adds a square magnitude on it, also known as L2 regularization.

Ridge regression may be used to treat collinearity issues (although stochastic and numerical approximation methods may solve this in linear regression)
and also to achieve a more generalized (better fitting) model as the _lambda_ parameters accounts for an addition in bias which in turn can decrease the
overall quadratic error (by reducing variance).

It has been seen that very high values in beta coefficients, as result of linear regression, often as result of high collinearity, can be avoided
by using a Ridge regression, penalizing higher values of beta coefficients.



### Parameters

<div class="table-striped">
  <table class="table">
    <tr>
        <th>Parameter</th>
        <th>Description</th>
        <th>Default Value</th>
     </tr>
     <tr>
        <td><code>'lambda</code></td>
        <td>Regularization parameter for Ridge Regression (the larger, the more generalized the model is)</td>
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

```scala
import org.apache.mahout.math.algorithms.regression.RidgeRegressionModel
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

val model = new RidgeRegression().fit(drmX, drmY, 'lambda -> 1.0)
val myAnswer = model.predict(drmX).collect
```
