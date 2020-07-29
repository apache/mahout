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
title: MeanCenter

    
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



