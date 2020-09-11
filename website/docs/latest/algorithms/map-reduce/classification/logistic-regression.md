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
title: (Deprecated)  Logistic Regression

    
---

<a name="LogisticRegression-LogisticRegression(SGD)"></a>
# Logistic Regression (SGD)

Logistic regression is a model used for prediction of the probability of
occurrence of an event. It makes use of several predictor variables that
may be either numerical or categories.

Logistic regression is the standard industry workhorse that underlies many
production fraud detection and advertising quality and targeting products. 
The Mahout implementation uses Stochastic Gradient Descent (SGD) to all
large training sets to be used.

For a more detailed analysis of the approach, have a look at the [thesis of
Paul Komarek](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1221&context=robotics) [1].

See MAHOUT-228 for the main JIRA issue for SGD.

A more detailed overview of the Mahout Linear Regression classifier and [detailed discription of building a Logistic Regression classifier](http://blog.trifork.com/2014/02/04/an-introduction-to-mahouts-logistic-regression-sgd-classifier/) for the classic [Iris flower dataset](http://en.wikipedia.org/wiki/Iris_flower_data_set) is also available [2]. 

An example of training a Logistic Regression classifier for the [UCI Bank Marketing Dataset](http://mlr.cs.umass.edu/ml/datasets/Bank+Marketing) can be found [on the Mahout website](http://mahout.apache.org/users/classification/bankmarketing-example.html) [3].

An example of training and testing a Logistic Regression document classifier for the classic [20 newsgroups corpus](https://github.com/apache/mahout/blob/master/examples/bin/classify-20newsgroups.sh) [4] is also available. 

<a name="LogisticRegression-Parallelizationstrategy"></a>
## Parallelization strategy

The bad news is that SGD is an inherently sequential algorithm.  The good
news is that it is blazingly fast and thus it is not a problem for Mahout's
implementation to handle training sets of tens of millions of examples. 
With the down-sampling typical in many data-sets, this is equivalent to a
dataset with billions of raw training examples.

The SGD system in Mahout is an online learning algorithm which means that
you can learn models in an incremental fashion and that you can do
performance testing as your system runs.  Often this means that you can
stop training when a model reaches a target level of performance.  The SGD
framework includes classes to do on-line evaluation using cross validation
(the CrossFoldLearner) and an evolutionary system to do learning
hyper-parameter optimization on the fly (the AdaptiveLogisticRegression). 
The AdaptiveLogisticRegression system makes heavy use of threads to
increase machine utilization.  The way it works is that it runs 20
CrossFoldLearners in separate threads, each with slightly different
learning parameters.  As better settings are found, these new settings are
propagating to the other learners.

<a name="LogisticRegression-Designofpackages"></a>
## Design of packages

There are three packages that are used in Mahout's SGD system.	These
include

* The vector encoding package (found in org.apache.mahout.vectorizer.encoders)

* The SGD learning package (found in org.apache.mahout.classifier.sgd)

* The evolutionary optimization system (found in org.apache.mahout.ep)

<a name="LogisticRegression-Featurevectorencoding"></a>
## Feature vector encoding

Because the SGD algorithms need to have fixed length feature vectors and
because it is a pain to build a dictionary ahead of time, most SGD
applications use the hashed feature vector encoding system that is rooted
at FeatureVectorEncoder.

The basic idea is that you create a vector, typically a
RandomAccessSparseVector, and then you use various feature encoders to
progressively add features to that vector.  The size of the vector should
be large enough to avoid feature collisions as features are hashed.

There are specialized encoders for a variety of data types.  You can
normally encode either a string representation of the value you want to
encode or you can encode a byte level representation to avoid string
conversion.  In the case of ContinuousValueEncoder and
ConstantValueEncoder, it is also possible to encode a null value and pass
the real value in as a weight.	This avoids numerical parsing entirely in
case you are getting your training data from a system like Avro.

Here is a class diagram for the encoders package:

![class diagram](../../images/vector-class-hierarchy.png)

<a name="LogisticRegression-SGDLearning"></a>
## SGD Learning

For the simplest applications, you can construct an
OnlineLogisticRegression and be off and running.  Typically, though, it is
nice to have running estimates of performance on held out data.  To do
that, you should use a CrossFoldLearner which keeps a stable of five (by
default) OnlineLogisticRegression objects.  Each time you pass a training
example to a CrossFoldLearner, it passes this example to all but one of its
children as training and passes the example to the last child to evaluate
current performance.  The children are used for evaluation in a round-robin
fashion so, if you are using the default 5 way split, all of the children
get 80% of the training data for training and get 20% of the data for
evaluation.

To avoid the pesky need to configure learning rates, regularization
parameters and annealing schedules, you can use the
AdaptiveLogisticRegression.  This class maintains a pool of
CrossFoldLearners and adapts learning rates and regularization on the fly
so that you don't have to.

Here is a class diagram for the classifiers.sgd package.  As you can see,
the number of twiddlable knobs is pretty large.  For some examples, see the
[TrainNewsGroups](https://github.com/apache/mahout/blob/master/examples/src/main/java/org/apache/mahout/classifier/sgd/TrainNewsGroups.java) example code.

![sgd class diagram](../../images/sgd-class-hierarchy.png)

## References

[1] [Thesis of
Paul Komarek](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1221&context=robotics)

[2] [An Introduction To Mahout's Logistic Regression SGD Classifier](http://blog.trifork.com/2014/02/04/an-introduction-to-mahouts-logistic-regression-sgd-classifier/)

## Examples

[3] [SGD Bank Marketing Example](http://mahout.apache.org/users/classification/bankmarketing-example.html)

[4] [SGD 20 newsgroups classification](https://github.com/apache/mahout/blob/master/examples/bin/classify-20newsgroups.sh)

