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
title: (Deprecated)  Latent Dirichlet Allocation

   
---

<a name="LatentDirichletAllocation-Overview"></a>
# Overview

Latent Dirichlet Allocation (Blei et al, 2003) is a powerful learning
algorithm for automatically and jointly clustering words into "topics" and
documents into mixtures of topics. It has been successfully applied to
model change in scientific fields over time (Griffiths and Steyvers, 2004;
Hall, et al. 2008). 

A topic model is, roughly, a hierarchical Bayesian model that associates
with each document a probability distribution over "topics", which are in
turn distributions over words. For instance, a topic in a collection of
newswire might include words about "sports", such as "baseball", "home
run", "player", and a document about steroid use in baseball might include
"sports", "drugs", and "politics". Note that the labels "sports", "drugs",
and "politics", are post-hoc labels assigned by a human, and that the
algorithm itself only assigns associate words with probabilities. The task
of parameter estimation in these models is to learn both what the topics
are, and which documents employ them in what proportions.

Another way to view a topic model is as a generalization of a mixture model
like [Dirichlet Process Clustering](http://en.wikipedia.org/wiki/Dirichlet_process)
. Starting from a normal mixture model, in which we have a single global
mixture of several distributions, we instead say that _each_ document has
its own mixture distribution over the globally shared mixture components.
Operationally in Dirichlet Process Clustering, each document has its own
latent variable drawn from a global mixture that specifies which model it
belongs to, while in LDA each word in each document has its own parameter
drawn from a document-wide mixture.

The idea is that we use a probabilistic mixture of a number of models that
we use to explain some observed data. Each observed data point is assumed
to have come from one of the models in the mixture, but we don't know
which.	The way we deal with that is to use a so-called latent parameter
which specifies which model each data point came from.

<a name="LatentDirichletAllocation-CollapsedVariationalBayes"></a>
# Collapsed Variational Bayes
The CVB algorithm which is implemented in Mahout for LDA combines
advantages of both regular Variational Bayes and Gibbs Sampling.  The
algorithm relies on modeling dependence of parameters on latest variables
which are in turn mutually independent.   The algorithm uses 2
methodologies to marginalize out parameters when calculating the joint
distribution and the other other is to model the posterior of theta and phi
given the inputs z and x.

A common solution to the CVB algorithm is to compute each expectation term
by using simple Gaussian approximation which is accurate and requires low
computational overhead.  The specifics behind the approximation involve
computing the sum of the means and variances of the individual Bernoulli
variables.

CVB with Gaussian approximation is implemented by tracking the mean and
variance and subtracting the mean and variance of the corresponding
Bernoulli variables.  The computational cost for the algorithm scales on
the order of O(K) with each update to q(z(i,j)).  Also for each
document/word pair only 1 copy of the variational posterior is required
over the latent variable.

<a name="LatentDirichletAllocation-InvocationandUsage"></a>
# Invocation and Usage

Mahout's implementation of LDA operates on a collection of SparseVectors of
word counts. These word counts should be non-negative integers, though
things will-- probably --work fine if you use non-negative reals. (Note
that the probabilistic model doesn't make sense if you do!) To create these
vectors, it's recommended that you follow the instructions in [Creating Vectors From Text](../basics/creating-vectors-from-text.html)
, making sure to use TF and not TFIDF as the scorer.

Invocation takes the form:


    bin/mahout cvb \
        -i <input path for document vectors> \
        -dict <path to term-dictionary file(s) , glob expression supported> \
        -o <output path for topic-term distributions>
        -dt <output path for doc-topic distributions> \
        -k <number of latent topics> \
        -nt <number of unique features defined by input document vectors> \
        -mt <path to store model state after each iteration> \
        -maxIter <max number of iterations> \
        -mipd <max number of iterations per doc for learning> \
        -a <smoothing for doc topic distributions> \
        -e <smoothing for term topic distributions> \
        -seed <random seed> \
        -tf <fraction of data to hold for testing> \
        -block <number of iterations per perplexity check, ignored unless
test_set_percentage>0> \


Topic smoothing should generally be about 50/K, where K is the number of
topics. The number of words in the vocabulary can be an upper bound, though
it shouldn't be too high (for memory concerns). 

Choosing the number of topics is more art than science, and it's
recommended that you try several values.

After running LDA you can obtain an output of the computed topics using the
LDAPrintTopics utility:


    bin/mahout ldatopics \
        -i <input vectors directory> \
        -d <input dictionary file> \
        -w <optional number of words to print> \
        -o <optional output working directory. Default is to console> \
        -h <print out help> \
        -dt <optional dictionary type (text|sequencefile). Default is text>



<a name="LatentDirichletAllocation-Example"></a>
# Example

An example is located in mahout/examples/bin/build-reuters.sh. The script
automatically downloads the Reuters-21578 corpus, builds a Lucene index and
converts the Lucene index to vectors. By uncommenting the last two lines
you can then cause it to run LDA on the vectors and finally print the
resultant topics to the console. 

To adapt the example yourself, you should note that Lucene has specialized
support for Reuters, and that building your own index will require some
adaptation. The rest should hopefully not differ too much.

<a name="LatentDirichletAllocation-ParameterEstimation"></a>
# Parameter Estimation

We use mean field variational inference to estimate the models. Variational
inference can be thought of as a generalization of [EM](expectation-maximization.html)
 for hierarchical Bayesian models. The E-Step takes the form of, for each
document, inferring the posterior probability of each topic for each word
in each document. We then take the sufficient statistics and emit them in
the form of (log) pseudo-counts for each word in each topic. The M-Step is
simply to sum these together and (log) normalize them so that we have a
distribution over the entire vocabulary of the corpus for each topic. 

In implementation, the E-Step is implemented in the Map, and the M-Step is
executed in the reduce step, with the final normalization happening as a
post-processing step.

<a name="LatentDirichletAllocation-References"></a>
# References

[David M. Blei, Andrew Y. Ng, Michael I. Jordan, John Lafferty. 2003. Latent Dirichlet Allocation. JMLR.](-http://machinelearning.wustl.edu/mlpapers/paper_files/BleiNJ03.pdf)

[Thomas L. Griffiths and Mark Steyvers. 2004. Finding scientific topics. PNAS.  ](http://psiexp.ss.uci.edu/research/papers/sciencetopics.pdf)

[David Hall, Dan Jurafsky, and Christopher D. Manning. 2008. Studying the History of Ideas Using Topic Models ](-http://aclweb.org/anthology//D/D08/D08-1038.pdf)
