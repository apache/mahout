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
title: (Deprecated) 

    
---

Notice:    Licensed to the Apache Software Foundation (ASF) under one
           or more contributor license agreements.  See the NOTICE file
           distributed with this work for additional information
           regarding copyright ownership.  The ASF licenses this file
           to you under the Apache License, Version 2.0 (the
           "License"); you may not use this file except in compliance
           with the License.  You may obtain a copy of the License at
           .
             http://www.apache.org/licenses/LICENSE-2.0
           .
           Unless required by applicable law or agreed to in writing,
           software distributed under the License is distributed on an
           "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
           KIND, either express or implied.  See the License for the
           specific language governing permissions and limitations
           under the License.

#Bank Marketing Example

### Introduction

This page describes how to run Mahout's SGD classifier on the [UCI Bank Marketing dataset](http://mlr.cs.umass.edu/ml/datasets/Bank+Marketing).
The goal is to predict if the client will subscribe a term deposit offered via a phone call. The features in the dataset consist
of information such as age, job, marital status as well as information about the last contacts from the bank.

### Code & Data

The bank marketing example code lives under 

*mahout-examples/src/main/java/org.apache.mahout.classifier.sgd.bankmarketing*

The data can be found at 

*mahout-examples/src/main/resources/bank-full.csv*

### Code details

This example consists of 3 classes:

  - BankMarketingClassificationMain
  - TelephoneCall
  - TelephoneCallParser

When you run the main method of BankMarketingClassificationMain it parses the dataset using the TelephoneCallParser and trains
a logistic regression model with 20 runs and 20 passes. The TelephoneCallParser uses Mahout's feature vector encoder
to encode the features in the dataset into a vector. Afterwards the model is tested and the learning rate and AUC is printed accuracy is printed to standard output.