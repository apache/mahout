---
layout: doc-page
title: AsFactor
redirect_from:
      - /docs/latest/algorithms/preprocessors/AsFactor
      - /docs/latest/algorithms/preprocessors/AsFactor.html
---


### About

The `AsFactor` preprocessor is used to turn the integer values of the columns into sparse vectors where the value is 1
 at the index that corresponds to the 'category' of that column.  This is also known as "One Hot Encoding" in many other
 packages. 
 

### Parameters

`AsFactor` takes no parameters.
 
### Example

```scala
import org.apache.mahout.math.algorithms.preprocessing.AsFactor

val A = drmParallelize(dense(
      (3, 2, 1, 2),
      (0, 0, 0, 0),
      (1, 1, 1, 1)), numPartitions = 2)

// 0 -> 2, 3 -> 5, 6 -> 9
val factorizer: AsFactorModel = new AsFactor().fit(A)

val factoredA = factorizer.transform(A)
```

