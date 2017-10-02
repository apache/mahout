---
layout: default
title: Release Notes
theme:
    name: retro-mahout
---

# Release Notes

#### 11 April 2016 - Apache Mahout 0.12.0 released

This release marks a major milestone for the “Samsara” environment’s goal
of providing an engine neutral math platform by now supporting Apache Flink.
While still experimental, the mahout Flink bindings now offer all of the R-Like
semantics for linear algebra operations, matrix decompositions, 
and algorithms of the “Samsara” platform for execution on a Flink back-end.

This release gives users of Apache Flink out of the box access to the following features (and more):

<ol>
<li>The Mahout Distributed Row Matrix (DRM) API.</li>
<li>Distributed and local Vector and Matrix algebra routines.</li>
<li>Distributed and local Stochastic Principal Component Analysis.</li>
<li>Distributed and local Stochastic Singular Value Decomposition.</li>
<li>Distributed and local Thin QR Decomposition.</li>
<li>Collaborative Filtering.</li>
<li>Naive Bayes Classification.</li>
<li>Matrix operations (only listing a few here):
<ol>
<li>Mahout-native blockified distributed Matrix map and allreduce routines.</li>
<li>Distributed data point (row) sampling.</li>
<li>Matrix/Matrix Squared Distance.</li>
<li>Element-wise log.</li>
<li>Element-wise roots.</li>
<li>Element-wise Matrix/Matrix addition, subtraction, division and multiplication.</li>
<li>Functional Matrix value assignment.</li>
<li>A familiar Scala-based R-like DSL.</li>
</ol>
</ol>

#### 11 March 2016 - Apache Mahout 0.11.2 released

This is a minor release over Mahout 0.11.1 meant to introduce major
performance enhancements with sparse matrix and vector computations, and
major performance optimizations to the Samsara DSL.  Mahout 0.11.2 includes
all new features and bug fixes released in Mahout versions 0.11.0 and
0.11.1.

Highlights include:

* Spark 1.5.2 support
*   Performance improvements of over 30% on Sparse Vector and Matrix
   computations leveraging the ‘fastutil’ library -  contribution from
   Sebastiano Vigna. This speeds up all in-core sparse vector and matrix
   computations.


#### 06 November 2015 - Apache Mahout 0.11.1 released

This is a minor release over Mahout 0.11.0 meant to expand Mahout’s
compatibility with Spark versions, to introduce some new features and to
fix some bugs.  Mahout 0.11.1 includes all new features and bug fixes
released in Mahout versions 0.11.0 and earlier.

Highlights include:

* Spark 1.4+ support
* 4x Performance improvement in Dot Product over Dense Vectors (https://issues.apache.org/jira/browse/MAHOUT-1781)


#### 07 August 2015 - Apache Mahout 0.11.0 released

Mahout 0.11.0 includes all new features and bugfixes released in Mahout versions 0.10.1
and 0.10.2 along with support for Spark 1.3+.

Highlights include:

* Spark 1.3 support
* Fixes for a major memory usage bug in co-occurrence analysis used by the driver spark-itemsimilarity. This will now require far less memory in the executor.
* Some minor fixes to Mahout-Samsara QR Decomposition and matrix ops.
* All of the Mahout Samsara fixes from 0.10.2 Release


#### 06 August 2015 - Apache Mahout 0.10.2 released

Highlights include:

* In-core transpose view rewrites. Modifiable transpose views eg. (for (col <- a.t) col := 5).
* Performance and parallelization improvements for AB', A'B, A'A spark physical operators.
* Optional structural "flavor" abstraction for in-core matrices. In-core matrices can now be tagged as e.g. sparse or dense.
* %*% optimization based on matrix flavors.
* In-core ::= sparse assignment functions.
* Assign := optimization (do proper traversal based on matrix flavors, similarly to %*%).
* Adding in-place elementwise functional assignment (e.g. mxA := exp _, mxA ::= exp _).
* Distributed and in-core version of simple elementwise analogues of scala.math._. for example, for log(x) the convention is dlog(drm), mlog(mx), vlog(vec). Unfortunately we cannot overload these functions over what is done in scala.math, i.e. scala would not allow log(mx) or log(drm) and log(Double) at the same time, mainly because they are being defined in different packages.
* Distributed and in-core first and second moment routines. R analogs: mean(), colMeans(), rowMeans(), variance(), sd(). By convention, distributed versions are prepended by (d) letter: colMeanVars() colMeanStdevs() dcolMeanVars() dcolMeanStdevs().
* Distance and squared distance matrix routines. R analog: dist(). Provide both squared and non-squared Euclidean distance matrices. By convention, distributed versions are prepended by (d) letter: dist(x), sqDist(x), dsqDist(x). Also a variation for pair-wise distance matrix of two different inputs x and y: sqDist(x,y), dsqDist(x,y). 
* DRM row sampling api.
* Distributed performance bug fixes. This relates mostly to (a) matrix multiplication deficiencies, and (b) handling parallelism.
* Distributed engine neutral allreduceBlock() operator api for Spark and H2O.
* Distributed optimizer operators for elementwise functions. Rewrites recognizing e.g. 1+ drmX * dexp(drmX) as a single fused elementwise physical operator: elementwiseFunc(f1(f2(drmX)) where f1 = 1 + x and f2 = exp(x).
* More cbind, rbind flavors (e.g. 1 cbind mxX, 1 cbind drmX or the other way around) for Spark and H2O.
* Added +=: and *=: operators on vectors.
* Closeable API for broadcast tensors.
* Support for conversion of any type-keyed DRM into ordinally-keyed DRM.
* Scala logging style. 
* rowSumsMap() summary for non-int-keyed DRMs.
* elementwise power operator ^ . 
* R-like vector concatenation operator. 
* In-core functional assignments e.g.: mxA := { (x) => x * x}. 
* Straighten out behavior of Matrix.iterator() and iterateNonEmpty().
* New mutable transposition view for in-core matrices.  In-core matrix transpose view. rewrite with mostly two goals in mind: (1) enable mutability, e.g. for (col <- mxA.t) col := k (2) translate matrix structural flavor for optimizers correctly. i.e. new SparseRowMatrix.t carries on as column-major structure.
* Native support for kryo serialization of tensor types.
* Deprecation of the MultiLayerPerceptron, ConcatenateVectorsJob and all related classes.
* Deprecation of SparseColumnMatrix.

#### 31 May 2015 - Apache Mahout 0.10.1 released

Highlights include:

* Major memory use improvements in  cooccurrence analysis including the spark-itemsimilarity driver [MAHOUT-1707](https://issues.apache.org/jira/browse/MAHOUT-1707)
* Support for Spark version 1.2.2 or less.
* Some minor fixes to Mahout-Samsara QR Decomposition and matrix ops.
* Trim down packages size to < 200MB MAHOUT-1704 and MAHOUT-1706
* Minor testing indicates binary compatibility with Spark 1.3 with the exception of the Mahout Shell.

#### 11 April 2015 - Apache Mahout 0.10.0 released

Mahout 0.10.0 was a major release, which separates out a ML environment (we call Mahout-Samsara) including an 
extended version of Scala that is largely backend independent but runs fully on Spark. The Hadoop MapReduce versions of 
Mahout algorithms are still maintained but no new MapReduce contributions are accepted. From this release onwards 
contributions must be Mahout Samsara based or at least run on Spark. 

Highlights include:

New Mahout Samsara Environment

* Distributed Algebraic optimizer
* R-Like DSL Scala API
* Linear algebra operations
* Ops are extensions to Scala
* Scala REPL based interactive shell running on Spark
* Integrates with compatible libraries like MLlib
* Run on distributed Spark
* H2O in progress

New Mahout Samsara based Algorithms

* Stochastic Singular Value Decomposition (ssvd, dssvd)
* Stochastic Principal Component Analysis (spca, dspca)
* Distributed Cholesky QR (thinQR)
* Distributed regularized Alternating Least Squares (dals)
* Collaborative Filtering: Item and Row Similarity
* Naive Bayes Classification
* Distributed and in-core

Changes in 0.10.0 are detailed <a href="https://github.com/apache/mahout/blob/mahout-0.10.0/CHANGELOG">here</a>

#### 1 February 2014 - Apache Mahout 0.9 released

  <p>Highlights include:</p>

  <ul>
   <li>New and improved Mahout website based on Apache CMS - <a href="https://issues.apache.org/jira/browse/MAHOUT-1245">MAHOUT-1245</a></li>
   <li>Early implementation of a Multi Layer Perceptron (MLP) classifier - <a href="https://issues.apache.org/jira/browse/MAHOUT-1265">MAHOUT-1265</a>.</li>
   <li>Scala DSL Bindings for Mahout Math Linear Algebra.  See <a href="http://weatheringthrutechdays.blogspot.com/2013/07/scala-dsl-for-mahout-in-core-linear.html">this blogpost</a> - <a href="https://issues.apache.org/jira/browse/MAHOUT-1297">MAHOUT-1297</a></li>
   <li>Recommenders as a Search.  See <a href="https://github.com/pferrel/solr-recommender">https://github.com/pferrel/solr-recommender</a> - <a href="https://issues.apache.org/jira/browse/MAHOUT-1288">MAHOUT-1288</a></li>
   <li>Support for easy functional Matrix views and derivatives - <a href="https://issues.apache.org/jira/browse/MAHOUT-1300">MAHOUT-1300</a></li>
   <li>JSON output format for ClusterDumper - <a href="https://issues.apache.org/jira/browse/MAHOUT-1343">MAHOUT-1343</a></li>
   <li>Enable randomised testing for all Mahout modules using Carrot RandomizedRunner - <a href="https://issues.apache.org/jira/browse/MAHOUT-1345">MAHOUT-1345</a></li>
   <li>Online Algorithm for computing accurate Quantiles using 1-dimensional Clustering - <a href="https://issues.apache.org/jira/browse/MAHOUT-1361">MAHOUT-1361</a>.  See this <a href="https://github.com/tdunning/t-digest/blob/master/docs/theory/t-digest-paper/histo.pdf">pdf</a> for the details.
   <li>Upgrade to Lucene 4.6.1 - <a href="https://issues.apache.org/jira/browse/MAHOUT-1364">MAHOUT-1364</a></li>
  </ul>

  <p>Changes in 0.9 are detailed <a href="http://svn.apache.org/viewvc/mahout/trunk/CHANGELOG?view=markup&pathrev=1563661">here</a>.</p>

#### 25 July 2013 - Apache Mahout 0.8 released

  <p>Highlights include:</p>

  <ul>
    <li>Numerous performance improvements to Vector and Matrix implementations, API's and their iterators</li>
		<li>Numerous performance improvements to the recommender implementations</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-1088" class="external-link" rel="nofollow">MAHOUT-1088</a>: Support for biased item-based recommender</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-1089" class="external-link" rel="nofollow">MAHOUT-1089</a>: SGD matrix factorization for rating prediction with user and item biases</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-1106" class="external-link" rel="nofollow">MAHOUT-1106</a>: Support for SVD++</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-944" class="external-link" rel="nofollow">MAHOUT-944</a>:  Support for converting one or more Lucene storage indexes to SequenceFiles as well as an upgrade of the supported Lucene version to Lucene 4.3.1.</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-1154" class="external-link" rel="nofollow">MAHOUT-1154</a> and friends: New streaming k-means implementation that offers on-line (and fast) clustering</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-833" class="external-link" rel="nofollow">MAHOUT-833</a>: Make conversion to SequenceFiles Map-Reduce, 'seqdirectory' can now be run as a MapReduce job.</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-1052" class="external-link" rel="nofollow">MAHOUT-1052</a>: Add an option to MinHashDriver that specifies the dimension of vector to hash (indexes or values).</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-884" class="external-link" rel="nofollow">MAHOUT-884</a>: Matrix Concat utility, presently only concatenates two matrices.</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-1187" class="external-link" rel="nofollow">MAHOUT-1187</a>: Upgraded to CommonsLang3</li>
		<li><a href="https://issues.apache.org/jira/browse/MAHOUT-916" class="external-link" rel="nofollow">MAHOUT-916</a>: Speedup the Mahout build by making tests run in parallel.</li>

  </ul>

  <p>Changes in 0.8 are detailed <a href="http://svn.apache.org/viewvc/mahout/trunk/CHANGELOG?revision=1501110&view=markup">here</a>.</p>

#### 16 June 2012 - Apache Mahout 0.7 released

  <p>Highlights include:</p>

  <ul>
    <li>Outlier removal capability in K-Means, Fuzzy K, Canopy and Dirichlet Clustering</li>
    <li>New Clustering implementation for K-Means, Fuzzy K, Canopy and Dirichlet using Cluster Classifiers</li>
    <li>Collections and Math API consolidated</li>
    <li>(Complementary) Naive Bayes refactored and cleaned</li>
    <li>Watchmaker and Old Naive Bayes dropped.</li>
    <li>Many bug fixes, refactorings, and other small improvements</li>
  </ul>

  <p>Changes in 0.7 are detailed <a href="https://issues.apache.org/jira/secure/ReleaseNote.jspa?projectId=12310751&version=12319261">here</a>.</p>



#### 6 Feb 2012 - Apache Mahout 0.6 released

  <p>Highlights include:</p>

  <ul>
    <li>Improved Decision Tree performance and added support for regression problems</li>
    <li>New LDA implementation using Collapsed Variational Bayes 0th Derivative Approximation</li>
    <li>Reduced runtime of LanczosSolver tests</li>
    <li>K-Trusses, Top-Down and Bottom-Up clustering, Random Walk with Restarts implementation</li>
    <li>Reduced runtime of dot product between vectors</li>
    <li>Added MongoDB and Cassandra DataModel support</li>
    <li>Increased efficiency of parallel ALS matrix factorization</li>
    <li>SSVD enhancements</li>
    <li>Performance improvements in RowSimilarityJob, TransposeJob</li>
    <li>Added numerous clustering display examples</li>
    <li>Many bug fixes, refactorings, and other small improvements</li>
  </ul>

  <p>Changes in 0.6 are detailed <a href="https://issues.apache.org/jira/secure/ReleaseNote.jspa?projectId=12310751&version=12316364">here</a>.</p>

#### Past Releases

 * [Mahout 0.5](https://issues.apache.org/jira/secure/ReleaseNote.jspa?version=12315255&styleName=Text&projectId=12310751&Create=Create&atl_token=A5KQ-2QAV-T4JA-FDED|20f0d06214912accbd47acf2f0a89231ed00a767|lin)
 * [Mahout 0.4](https://issues.apache.org/jira/secure/ReleaseNote.jspa?version=12314281&styleName=Text&projectId=12310751&Create=Create&atl_token=A5KQ-2QAV-T4JA-FDED|20f0d06214912accbd47acf2f0a89231ed00a767|lin)
 * [Mahout 0.3](https://issues.apache.org/jira/secure/ReleaseNote.jspa?version=12314281&styleName=Text&projectId=12310751&Create=Create&atl_token=A5KQ-2QAV-T4JA-FDED|20f0d06214912accbd47acf2f0a89231ed00a767|lin)
 * [Mahout 0.2](https://issues.apache.org/jira/secure/ReleaseNote.jspa?version=12313278&styleName=Text&projectId=12310751&Create=Create&atl_token=A5KQ-2QAV-T4JA-FDED|20f0d06214912accbd47acf2f0a89231ed00a767|lin) 
 * [Mahout 0.1](https://issues.apache.org/jira/secure/ReleaseNote.jspa?version=12312976&styleName=Html&projectId=12310751&Create=Create&atl_token=A5KQ-2QAV-T4JA-FDED%7C48e83cdefb8bca42acf8f129692f8c3a05b360cf%7Clout)