---
layout: doc-page
title: Ridge Regression
---

### About

The `Ridge` regressor in Mahout implements a _closed-form_ solution to Ridge Regression (https://en.wikipedia.org/wiki/Tikhonov_regularization).



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
