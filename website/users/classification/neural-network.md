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
layout: default
title: Neural Network

    
---

<a name="NeuralNetwork-NeuralNetworks"></a>
# Neural Networks

Neural Networks are a means for classifying multi dimensional objects. We
concentrate on implementing back propagation networks with one hidden layer
as these networks have been covered by the [2006 NIPS map reduce paper](http://www.cs.stanford.edu/people/ang/papers/nips06-mapreducemulticore.pdf)
. Those networks are capable of learning not only linear separating hyper
planes but arbitrary decision boundaries.

<a name="NeuralNetwork-Strategyforparallelbackpropagationnetwork"></a>
## Strategy for parallel backpropagation network


<a name="NeuralNetwork-Designofimplementation"></a>
## Design of implementation
