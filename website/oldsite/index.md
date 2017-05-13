---
layout: default
title: Downloads
theme:
    name: retro-mahout
---

# What is Apache Mahout?
## The Apache Mahout™ project's goal is to build an environment for quickly creating scalable performant machine learning applications.

  <div class="highlights">
    <a href="http://mahout.apache.org/general/downloads.html"><img src="http://mahout.apache.org/images/download-mahout.png"/></a>
    <h4>Latest release version 0.12.2 has</h4>
    <h6>Apache Mahout Samsara Environment includes</h6>
    <ul>
      <li>Distributed Algebraic optimizer</li>
      <li>R-Like DSL Scala API</li>
      <li>Linear algebra operations</li>
      <li>Ops are extensions to Scala</li>
      <li>IScala REPL based interactive shell</li>
      <li>Integrates with compatible libraries like MLLib</li>
      <li>Runs on distributed Spark, H2O, and Flink</li>
      <li>fastutil to speed up sparse matrix and vector computations</li>
      <li>Matrix to tsv conversions for integration with Apache Zeppelin</li>
    </ul>
    <h6>Apache Mahout Samsara Algorithms included</h6>
    <ul>
      <li>Stochastic Singular Value Decomposition (ssvd, dssvd)</li>
      <li>Stochastic Principal Component Analysis (spca, dspca)</li>
      <li>Distributed Cholesky QR (thinQR) </li>
      <li>Distributed regularized Alternating Least Squares (dals)</li>
      <li>Collaborative Filtering: Item and Row Similarity</li>
      <li>Naive Bayes Classification</li>
      <li>Distributed and in-core</li>
    </ul>
  </div>

**Apache Mahout software** provides three major features:
- A simple and extensible programming environment and framework for building scalable algorithms
- A wide variety of premade algorithms for Scala + Apache Spark, H2O, Apache Flink
- Samsara, a vector math experimentation environment with R-like syntax which works at scale


Read an [Overview of programming a Mahout Samsara Application][1], 
learn [How To Contribute to Mahout][2],
[report an issue, bug, or suggestion][3] in our JIRA, 
see the [Samsara bindings for Scala and Spark][4],
and [contact us on our mailing lists][5].

#### 13 May 2017 - Apache Mahout website beta release

Docs available [here](http://mahout.apache.org/docs/0.13.1-SNAPSHOT)

#### 17 April 2017 - Apache Mahout 0.13.0 released

Mahout 0.13.0 includes 

* In-core matrices backed by ViennaCL [3] providing in some cases speedups
of an order of magnitude.
* A JavaCPP bridge to native/GPU operations in ViennaCL
* Distributed GPU Matrix-Matrix and Matrix-Vector multiplication on Spark
* Distributed OpenMP Matrix-Matrix and Matrix-Vector multiplication on Spark
* Sparse and dense matrix GPU-backed support.
* Fault tolerance by falling back to Mahout JVM counterpart of new solvers
in the case of failure on GPU or OpenMP
* A new scikit-learn-like framework for algorithms with the goal for
creating a consistent API for various machine-learning algorithms and an
orderly package structure for grouping regression, classification,
clustering, and pre-processing algorithms together
* New DRM wrappers in Spark Bindings making it more convenient to create
DRMs from MLLib RDDs and DataFrames
* MahoutConversions adds Scala-like compatibility to Vectors introducing
methods such as toArray() and toMap()

#### 13 Jun 2016 - Apache Mahout 0.12.2 released

Mahout 0.12.2 is a minor release following 0.12.1 that provides Matrix exports for Apache Zeppelin integration along with a few minor bug fixes and dependency upgrades. 

 
#### 19 May 2016 - "Visualizing Apache Mahout in R via Apache Zeppelin" posted
<p>A tutorial on connecting Mahout, R, Spark, and Zeppelin by <a href="https://trevorgrant.org/2016/05/19/visualizing-apache-mahout-in-r-via-apache-zeppelin-incubating" target="_blank">Trevor Grant</a> showing how to plot results in Apache Zeppelin.</p>

#### 18 May 2016 - Apache Mahout 0.12.1 released 

Mahout 0.12.1 is a minor release following 0.12.0 that fixes issues in the newly added Flink backend and supports Flink 1.0.3.


#### **11 Apr 2016 - Apache Mahout 0.12.0 released**

The Mahout 0.12.0 release marks a major milestone for the “Samsara” environment’s goal of providing an engine neutral math platform by now supporting Flink.  While still experimental, the mahout Flink bindings now offer all of the R-Like semantics for linear algebra operations, matrix decompositions, and algorithms of the “Samsara” platform for execution on a Flink back-end.

 
#### **12 Mar 2016 - Apache Mahout 0.11.2 released**
**Apache Mahout introduces a new math environment called** [**Samsara**](http://mahout.apache.org/users/sparkbindings/home.html), 
    for its theme of universal renewal. It reflects a fundamental rethinking of how scalable machine learning 
    algorithms are built and customized. Mahout-Samsara is here to help people create their own math while providing
    some off-the-shelf algorithm implementations. At its core are general linear algebra and statistical operations 
   along with the data structures to support them. You can use is as a library or customize it in Scala with Mahout-specific extensions 
   that look something like R. 
   Mahout-Samsara comes with an interactive shell that runs distributed operations on an Apache Spark cluster. 
   This make prototyping or task submission much easier and allows users to customize algorithms with
   a whole new degree of freedom.

  [**Mahout Algorithms**](http://mahout.apache.org/users/basics/algorithms.html) include many new 
    implementations built for speed on Mahout-Samsara. They run on Spark 1.3+, Flink 1.0.1, and some on H2O, which means as 
    much as a 10x speed increase. You’ll find robust matrix decomposition algorithms as well as a **[Naive Bayes][6]** 
   classifier and collaborative filtering. The new spark-itemsimilarity enables the next generation of **[cooccurrence 
   recommenders][7]** that can use entire user click streams and context in making recommendations.

  Our [release notes](http://mahout.apache.org/general/release-notes.html) have details. Interested in helping? Join the <a href="https://mahout.apache.org/general/mailing-lists,-irc-and-archives.html">Mailing lists</a>.

# Mahout News

#### 13 May 2017 - Apache Mahout website beta release

Docs available [here](http://mahout.apache.org/docs/0.13.1-SNAPSHOT)

#### 17 April 2017 - Apache Mahout 0.13.0 released

#### 13 June 2016 - Apache Mahout 0.12.2 released

#### 18 May 2016 - Apache Mahout 0.12.1 released

#### 11 April 2016 - Apache Mahout 0.12.0 released

#### 12 March 2016 - Apache Mahout 0.11.2 released

#### 23 February 2016 - New Apache Mahout Book - "Apache Mahout: Beyond MapReduce" by D.Lyubimov and A.Palumbo released. See [link](http://www.weatheringthroughtechdays.com/2016/02/mahout-samsara-book-is-out.html) Mahout "Samsara" Book Is Out

#### 6 November 2015 - Apache Mahout 0.11.1 released

#### 7 August 2015 - Apache Mahout 0.11.0 released

#### 6 August 2015 - Apache Mahout 0.10.2 released

#### 31 May 2015 - Apache Mahout 0.10.1 released

#### 11 April 2015 - Apache Mahout 0.10.0 released

#### 1 February 2014 - Apache Mahout 0.9 released

Visit our [release notes](http://svn.apache.org/viewvc/mahout/trunk/CHANGELOG?view=markup&pathrev=1563661) page for details.



####25 July 2013 - Apache Mahout 0.8 released

Visit our [release notes](http://mahout.apache.org/general/release-notes.html) page for details.

#### 16 June 2012 - Apache Mahout 0.7 released

Visit our [release notes](http://mahout.apache.org/general/release-notes.html) page for details.

#### 6 Feb 2012 - Apache Mahout 0.6 released

Visit our [release notes](http://mahout.apache.org/general/release-notes.html) page for details.

#### 9 Oct 2011 - Mahout in Action released

The book *Mahout in Action* is available in print. Sean Owen, Robin Anil, Ted Dunning and Ellen Friedman thank the community (especially those who were reviewers) for input during the process and hope it is enjoyable.

Find *Mahout in Action* at your favorite bookstore, or [order print and eBook copies from Manning](http://manning.com/owen/) -- use discount code "mahout37" for 37% off.


  [1]: http://mahout.apache.org/users/environment/how-to-build-an-app.html
  [2]: http://mahout.apache.org/developers/how-to-contribute.html
  [3]: http://mahout.apache.org/developers/issue-tracker.html
  [4]: http://mahout.apache.org/users/sparkbindings/home.html
  [5]: http://mahout.apache.org/general/mailing-lists,-irc-and-archives.html
  [6]: http://mahout.apache.org/users/algorithms/spark-naive-bayes.html
  [7]: http://mahout.apache.org/users/algorithms/intro-cooccurrence-spark.html