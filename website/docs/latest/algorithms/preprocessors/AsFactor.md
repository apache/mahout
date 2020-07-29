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
title: AsFactor

    
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

