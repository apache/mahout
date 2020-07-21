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
title: (Deprecated)  Breiman Example

    
---

#Breiman Example

#### Introduction

This page describes how to run the Breiman example, which implements the test procedure described in [Leo Breiman's paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.3999&rep=rep1&type=pdf). The basic algorithm is as follows :

 * repeat *I* iterations
 * in each iteration do
  * keep 10% of the dataset apart as a testing set 
  * build two forests using the training set, one with *m = int(log2(M) + 1)* (called Random-Input) and one with *m = 1* (called Single-Input)
  * choose the forest that gave the lowest oob error estimation to compute
the test set error
  * compute the test set error using the Single Input Forest (test error),
this demonstrates that even with *m = 1*, Decision Forests give comparable
results to greater values of *m*
  * compute the mean testset error using every tree of the chosen forest
(tree error). This should indicate how well a single Decision Tree performs
 * compute the mean test error for all iterations
 * compute the mean tree error for all iterations


#### Running the Example

The current implementation is compatible with the [UCI repository](http://archive.ics.uci.edu/ml/) file format. We'll show how to run this example on two datasets:

First, we deal with [Glass Identification](http://archive.ics.uci.edu/ml/datasets/Glass+Identification): download the [dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data) file called **glass.data** and store it onto your local machine. Next, we must generate the descriptor file **glass.info** for this dataset with the following command:

    bin/mahout org.apache.mahout.classifier.df.tools.Describe -p /path/to/glass.data -f /path/to/glass.info -d I 9 N L

Substitute */path/to/* with the folder where you downloaded the dataset, the argument "I 9 N L" indicates the nature of the variables. Here it means 1
ignored (I) attribute, followed by 9 numerical(N) attributes, followed by
the label (L).

Finally, we build and evaluate our random forest classifier as follows:

    bin/mahout org.apache.mahout.classifier.df.BreimanExample -d /path/to/glass.data -ds /path/to/glass.info -i 10 -t 100
which builds 100 trees (-t argument) and repeats the test 10 iterations (-i
argument) 

The example outputs the following results:

 * Selection error: mean test error for the selected forest on all iterations
 * Single Input error: mean test error for the single input forest on all
iterations
 * One Tree error: mean single tree error on all iterations
 * Mean Random Input Time: mean build time for random input forests on all
iterations
 * Mean Single Input Time: mean build time for single input forests on all
iterations

We can repeat this for a [Sonar](http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar,+Mines+vs.+Rocks%29) usecase: download the [dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data) file called **sonar.all-data** and store it onto your local machine. Generate the descriptor file **sonar.info** for this dataset with the following command:

    bin/mahout org.apache.mahout.classifier.df.tools.Describe -p /path/to/sonar.all-data -f /path/to/sonar.info -d 60 N L

The argument "60 N L" means 60 numerical(N) attributes, followed by the label (L). Analogous to the previous case, we run the evaluation as follows:

    bin/mahout org.apache.mahout.classifier.df.BreimanExample -d /path/to/sonar.all-data -ds /path/to/sonar.info -i 10 -t 100



