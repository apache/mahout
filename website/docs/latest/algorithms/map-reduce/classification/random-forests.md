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
title: (Deprecated)  Random Forests

    
---

<a name="RandomForests-HowtogrowaDecisionTree"></a>
### How to grow a Decision Tree

source : \[3\](3\.html)

LearnUnprunedTree(*X*,*Y*)

Input: *X* a matrix of *R* rows and *M* columns where *X{*}{*}{~}ij{~}* =
the value of the *j*'th attribute in the *i*'th input datapoint. Each
column consists of either all real values or all categorical values.
Input: *Y* a vector of *R* elements, where *Y{*}{*}{~}i{~}* = the output
class of the *i*'th datapoint. The *Y{*}{*}{~}i{~}* values are categorical.
Output: An Unpruned decision tree


If all records in *X* have identical values in all their attributes (this
includes the case where *R<2*), return a Leaf Node predicting the majority
output, breaking ties randomly. This case also includes
If all values in *Y* are the same, return a Leaf Node predicting this value
as the output
Else
&nbsp;&nbsp;&nbsp; select *m* variables at random out of the *M* variables
&nbsp;&nbsp;&nbsp; For *j* = 1 .. *m*
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If *j*'th attribute is
categorical
*&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
IG{*}{*}{~}j{~}* = IG(*Y*\|*X{*}{*}{~}j{~}*) (see Information
Gain)&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Else (*j*'th attribute is
real-valued)
*&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
IG{*}{*}{~}j{~}* = IG*(*Y*\|*X{*}{*}{~}j{~}*) (see Information Gain)
&nbsp;&nbsp;&nbsp; Let *j\** = argmax{~}j~ *IG{*}{*}{~}j{~}* (this is the
splitting attribute we'll use)
&nbsp;&nbsp;&nbsp; If *j\** is categorical then
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For each value *v* of the *j*'th
attribute
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let
*X{*}{*}{^}v{^}* = subset of rows of *X* in which *X{*}{*}{~}ij{~}* = *v*.
Let *Y{*}{*}{^}v{^}* = corresponding subset of *Y*
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Let *Child{*}{*}{^}v{^}* =
LearnUnprunedTree(*X{*}{*}{^}v{^}*,*Y{*}{*}{^}v{^}*)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Return a decision tree node,
splitting on *j*'th attribute. The number of children equals the number of
values of the *j*'th attribute, and the *v*'th child is
*Child{*}{*}{^}v{^}*
&nbsp;&nbsp;&nbsp; Else *j\** is real-valued and let *t* be the best split
threshold
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let *X{*}{*}{^}LO{^}* = subset
of rows of *X* in which *X{*}{*}{~}ij{~}* *<= t*. Let *Y{*}{*}{^}LO{^}* =
corresponding subset of *Y*
&nbsp; &nbsp; &nbsp; &nbsp; Let *Child{*}{*}{^}LO{^}* =
LearnUnprunedTree(*X{*}{*}{^}LO{^}*,*Y{*}{*}{^}LO{^}*)
&nbsp; &nbsp; &nbsp; &nbsp; Let *X{*}{*}{^}HI{^}* = subset of rows of *X*
in which *X{*}{*}{~}ij{~}* *> t*. Let *Y{*}{*}{^}HI{^}* = corresponding
subset of *Y*
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let *Child{*}{*}{^}HI{^}* =
LearnUnprunedTree(*X{*}{*}{^}HI{^}*,*Y{*}{*}{^}HI{^}*)
&nbsp; &nbsp; &nbsp; &nbsp; Return a decision tree node, splitting on
*j*'th attribute. It has two children corresponding to whether the *j*'th
attribute is above or below the given threshold.

*Note*: There are alternatives to Information Gain for splitting nodes
&nbsp;

<a name="RandomForests-Informationgain"></a>
### Information gain

source : \[3\](3\.html)
1. h4. nominal attributes

suppose X can have one of m values V{~}1~,V{~}2~,...,V{~}m~
P(X=V{~}1~)=p{~}1~, P(X=V{~}2~)=p{~}2~,...,P(X=V{~}m~)=p{~}m~
&nbsp;
H(X)= \-sum{~}j=1{~}{^}m^ p{~}j~ log{~}2~ p{~}j~ (The entropy of X)
H(Y\|X=v) = the entropy of Y among only those records in which X has value
v
H(Y\|X) = sum{~}j~ p{~}j~ H(Y\|X=v{~}j~)
IG(Y\|X) = H(Y) - H(Y\|X)
1. h4. real-valued attributes

suppose X is real valued
define IG(Y\|X:t) as H(Y) - H(Y\|X:t)
define H(Y\|X:t) = H(Y\|X<t) P(X<t) + H(Y\|X>=t) P(X>=t)
define IG*(Y\|X) = max{~}t~ IG(Y\|X:t)

<a name="RandomForests-HowtogrowaRandomForest"></a>
### How to grow a Random Forest

source : \[1\](1\.html)

Each tree is grown as follows:
1. if the number of cases in the training set is *N*, sample *N* cases at
random \-but with replacement, from the original data. This sample will be
the training set for the growing tree.
1. if there are *M* input variables, a number *m << M* is specified such
that at each node, *m* variables are selected at random out of the *M* and
the best split on these *m* is used to split the node. The value of *m* is
held constant during the forest growing.
1. each tree is grown to its large extent possible. There is no pruning.

<a name="RandomForests-RandomForestparameters"></a>
### Random Forest parameters

source : \[2\](2\.html)
Random Forests are easy to use, the only 2 parameters a user of the
technique has to determine are the number of trees to be used and the
number of variables (*m*) to be randomly selected from the available set of
variables.
Breinman's recommendations are to pick a large number of trees, as well as
the square root of the number of variables for *m*.
&nbsp;

<a name="RandomForests-Howtopredictthelabelofacase"></a>
### How to predict the label of a case

Classify(*node*,*V*)
&nbsp;&nbsp;&nbsp; Input: *node* from the decision tree, if *node.attribute
= j* then the split is done on the *j*'th attribute

&nbsp;&nbsp; &nbsp;Input: *V* a vector of *M* columns where
*V{*}{*}{~}j{~}* = the value of the *j*'th attribute.
&nbsp;&nbsp;&nbsp; Output: label of *V*

&nbsp;&nbsp;&nbsp; If *node* is a Leaf then
&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; Return the value predicted
by *node*

&nbsp;&nbsp; &nbsp;Else
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Let *j =
node.attribute*
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If *j* is
categorical then
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Let *v* = *V{*}{*}{~}j{~}*
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Let *child{*}{*}{^}v{^}* = child node corresponding to the attribute's
value *v*
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp; Return Classify(*child{*}{*}{^}v{^}*,*V*)

&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Else *j* is
real-valued
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Let *t = node.threshold* (split threshold)
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp; If Vj < t then
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; Let *child{*}{*}{^}LO{^}* = child
node corresponding to (*<t*)
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; Return
Classify(*child{*}{*}{^}LO{^}*,*V*)
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Else
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; Let *child{*}{*}{^}HI{^}* =
child node corresponding to (*>=t*)
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; Return
Classify(*child{*}{*}{^}HI{^}*,*V*)
&nbsp;

<a name="RandomForests-Theoutofbag(oob)errorestimation"></a>
### The out of bag (oob) error estimation

source : \[1\](1\.html)

in random forests, there is no need for cross-validation or a separate test
set to get an unbiased estimate of the test set error. It is estimated
internally, during the run, as follows:
* each tree is constructed using a different bootstrap sample from the
original data. About one-third of the cases left of the bootstrap sample
and not used in the construction of the _kth_ tree.
* put each case left out in the construction of the _kth_ tree down the
_kth{_}tree to get a classification. In this way, a test set classification
is obtained for each case in about one-thrid of the trees. At the end of
the run, take *j* to be the class that got most of the the votes every time
case *n* was _oob_. The proportion of times that *j* is not equal to the
true class of *n* averaged over all cases is the _oob error estimate_. This
has proven to be unbiased in many tests.

<a name="RandomForests-OtherRFuses"></a>
### Other RF uses

source : \[1\](1\.html)
* variable importance
* gini importance
* proximities
* scaling
* prototypes
* missing values replacement for the training set
* missing values replacement for the test set
* detecting mislabeled cases
* detecting outliers
* detecting novelties
* unsupervised learning
* balancing prediction error
Please refer to \[1\](1\.html)
 for a detailed description

<a name="RandomForests-References"></a>
### References

\[1\](1\.html)
&nbsp; Random Forests - Classification Description
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;[http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
\[2\](2\.html)
&nbsp; B. Larivi�re & D. Van Den Poel, 2004. "Predicting Customer Retention
and Profitability by Using Random Forests and Regression Forests
Techniques,"
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Working Papers of Faculty of
Economics and Business Administration, Ghent University, Belgium 04/282,
Ghent University,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Faculty of Economics and
Business Administration.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Available online : [http://ideas.repec.org/p/rug/rugwps/04-282.html](http://ideas.repec.org/p/rug/rugwps/04-282.html)
\[3\](3\.html)
&nbsp; Decision Trees - Andrew W. Moore\[4\]
&nbsp; &nbsp; &nbsp; &nbsp; http://www.cs.cmu.edu/~awm/tutorials\[1\](1\.html)
\[4\](4\.html)
&nbsp; Information Gain - Andrew W. Moore
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [http://www.cs.cmu.edu/~awm/tutorials](http://www.cs.cmu.edu/~awm/tutorials)
