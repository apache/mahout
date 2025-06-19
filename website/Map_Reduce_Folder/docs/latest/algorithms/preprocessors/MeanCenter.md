---
layout: doc-page
title: MeanCenter
redirect_from:
      - /docs/latest/algorithms/preprocessors/MeanCenter
      - /docs/latest/algorithms/preprocessors/MeanCenter.html
---

### About

`MeanCenter` centers values about the column mean. 

### Parameters

### Example

```scala
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter

val A = drmParallelize(dense(
      (1, 1, -2),
      (2, 5, 2),
      (3, 9, 0)), numPartitions = 2)

val scaler: MeanCenterModel = new MeanCenter().fit(A)

val centeredA = scaler.transform(A)
```



