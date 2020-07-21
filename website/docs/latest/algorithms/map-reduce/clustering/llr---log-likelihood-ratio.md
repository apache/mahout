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
title: (Deprecated)  LLR - Log-likelihood Ratio

   
---

# Likelihood ratio test

_Likelihood ratio test is used to compare the fit of two models one
of which is nested within the other._

In the context of machine learning and the Mahout project in particular,
the term LLR is usually meant to refer to a test of significance for two
binomial distributions, also known as the G squared statistic.	This is a
special case of the multinomial test and is closely related to mutual
information.  The value of this statistic is not normally used in this
context as a true frequentist test of significance since there would be
obvious and dreadful problems to do with multiple comparisons, but rather
as a heuristic score to order pairs of items with the most interestingly
connected items having higher scores.  In this usage, the LLR has proven
very useful for discriminating pairs of features that have interesting
degrees of cooccurrence and those that do not with usefully small false
positive and false negative rates.  The LLR is typically far more suitable
in the case of small than many other measures such as Pearson's
correlation, Pearson's chi squared statistic or z statistics.  The LLR as
stated does not, however, make any use of rating data which can limit its
applicability in problems such as the Netflix competition. 

The actual value of the LLR is not usually very helpful other than as a way
of ordering pairs of items.  As such, it is often used to determine a
sparse set of coefficients to be estimated by other means such as TF-IDF. 
Since the actual estimation of these coefficients can be done in a way that
is independent of the training data such as by general corpus statistics,
and since the ordering imposed by the LLR is relatively robust to counting
fluctuation, this technique can provide very strong results in very sparse
problems where the potential number of features vastly out-numbers the
number of training examples and where features are highly interdependent.

 See Also: 

* [Blog post "surprise and coincidence"](http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html)
* [G-Test](http://en.wikipedia.org/wiki/G-test)
* [Likelihood Ratio Test](http://en.wikipedia.org/wiki/Likelihood-ratio_test)

      