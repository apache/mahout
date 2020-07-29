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

title: Regression Algorithms

    
---

Apache Mahout implements the following regression algorithms "off the shelf".

### Closed Form Solutions

These methods used close form solutions (not stochastic) to solve regression problems

[Ordinary Least Squares](ols.html)

### Autocorrelation Regression

Serial Correlation of the error terms can lead to biased estimates of regression parameters, the following remedial procedures are provided:

[Cochrane Orcutt Procedure](serial-correlation/cochrane-orcutt.html)

[Durbin Watson Test](serial-correlation/dw-test.html)
