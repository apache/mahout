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
title: (Deprecated)  Perceptron and Winnow

    
---
<a name="PerceptronandWinnow-ClassificationwithPerceptronorWinnow"></a>
# Classification with Perceptron or Winnow

Both algorithms are comparably simple linear classifiers. Given training
data in some n-dimensional vector space that is annotated with binary
labels the algorithms are guaranteed to find a linear separating hyperplane
if one exists. In contrast to the Perceptron, Winnow works only for binary
feature vectors.

For more information on the Perceptron see for instance:
http://en.wikipedia.org/wiki/Perceptron

Concise course notes on both algorithms:
http://pages.cs.wisc.edu/~shuchi/courses/787-F07/scribe-notes/lecture24.pdf

Although the algorithms are comparably simple they still work pretty well
for text classification and are fast to train even for huge example sets.
In contrast to Naive Bayes they are not based on the assumption that all
features (in the domain of text classification: all terms in a document)
are independent.

<a name="PerceptronandWinnow-Strategyforparallelisation"></a>
## Strategy for parallelisation

Currently the strategy for parallelisation is simple: Given there is enough
training data, split the training data. Train the classifier on each split.
The resulting hyperplanes are then averaged.

<a name="PerceptronandWinnow-Roadmap"></a>
## Roadmap

Currently the patch only contains the code for the classifier itself. It is
planned to provide unit tests and at least one example based on the WebKB
dataset by the end of November for the serial version. After that the
parallelisation will be added.
