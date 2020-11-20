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
title: Dimensional Reduction

   
---

# Support for dimensional reduction

Matrix algebra underpins the way many Big Data algorithms and data
structures are composed: full-text search can be viewed as doing matrix
multiplication of the term-document matrix by the query vector (giving a
vector over documents where the components are the relevance score),
computing co-occurrences in a collaborative filtering context (people who
viewed X also viewed Y, or ratings-based CF like the Netflix Prize contest)
is taking the squaring the user-item interaction matrix, calculating users
who are k-degrees separated from each other in a social network or
web-graph can be found by looking at the k-fold product of the graph
adjacency matrix, and the list goes on (and these are all cases where the
linear structure of the matrix is preserved!)

Each of these examples deal with cases of matrices which tend to be
tremendously large (often millions to tens of millions to hundreds of
millions of rows or more, by sometimes a comparable number of columns), but
also rather sparse. Sparse matrices are nice in some respects: dense
matrices which are 10^7 on a side would have 100 trillion non-zero entries!
But the sparsity is often problematic, because any given two rows (or
columns) of the matrix may have zero overlap. Additionally, any
machine-learning work done on the data which comprises the rows has to deal
with what is known as "the curse of dimensionality", and for example, there
are too many columns to train most regression or classification problems on
them independently.

One of the more useful approaches to dealing with such huge sparse data
sets is the concept of dimensionality reduction, where a lower dimensional
space of the original column (feature) space of your data is found /
constructed, and your rows are mapped into that subspace (or sub-manifold).
 In this reduced dimensional space, "important" components to distance
between points are exaggerated, and unimportant ones washed away, and
additionally, sparsity of your rows is traded for drastically reduced
dimensional, but dense "signatures". While this loss of sparsity can lead
to its own complications, a proper dimensionality reduction can help reveal
the most important features of your data, expose correlations among your
supposedly independent original variables, and smooth over the zeroes in
your correlation matrix.

One of the most straightforward techniques for dimensionality reduction is
the matrix decomposition: singular value decomposition, eigen
decomposition, non-negative matrix factorization, etc. In their truncated
form these decompositions are an excellent first approach toward linearity
preserving unsupervised feature selection and dimensional reduction. Of
course, sparse matrices which don't fit in RAM need special treatment as
far as decomposition is concerned. Parallelizable and/or stream-oriented
algorithms are needed.

<a name="DimensionalReduction-SingularValueDecomposition"></a>
# Singular Value Decomposition

Currently implemented in Mahout (as of 0.3, the first release with MAHOUT-180 applied), are two scalable implementations of SVD, a stream-oriented implementation using the Asymmetric Generalized Hebbian Algorithm outlined in Genevieve Gorrell & Brandyn Webb's paper ([Gorrell and Webb 2005](-http://www.dcs.shef.ac.uk/~genevieve/gorrell_webb.pdf.html)
); and there is a [Lanczos | http://en.wikipedia.org/wiki/Lanczos_algorithm]
 implementation, both single-threaded, and in the
o.a.m.math.decomposer.lanczos package (math module), as a hadoop map-reduce
(series of) job(s) in o.a.m.math.hadoop.decomposer package (core module).
Coming soon: stochastic decomposition.

See also: [https://cwiki.apache.org/confluence/display/MAHOUT/SVD+-+Singular+Value+Decomposition](Wikipedia - SVD)

<a name="DimensionalReduction-Lanczos"></a>
## Lanczos

The Lanczos algorithm is designed for eigen-decomposition, but like any
such algorithm, getting singular vectors out of it is immediate (singular
vectors of matrix A are just the eigenvectors of A^t * A or A * A^t). 
Lanczos works by taking a starting seed vector *v* (with cardinality equal
to the number of columns of the matrix A), and repeatedly multiplying A by
the result: *v'* = A.times(*v*) (and then subtracting off what is
proportional to previous *v'*'s, and building up an auxiliary matrix of
projections).  In the case where A is not square (in general: not
symmetric), then you actually want to repeatedly multiply A*A^t by *v*:
*v'* = (A * A^t).times(*v*), or equivalently, in Mahout,
A.timesSquared(*v*) (timesSquared is merely an optimization: by changing
the order of summation in A*A^t.times(*v*), you can do the same computation
as one pass over the rows of A instead of two).

After *k* iterations of *v_i* = A.timesSquared(*v_(i-1)*), a *k*- by -*k*
tridiagonal matrix has been created (the auxiliary matrix mentioned above),
out of which a good (often extremely good) approximation to *k* of the
singular values (and with the basis spanned by the *v_i*, the *k* singular
*vectors* may also be extracted) of A may be efficiently extracted.  Which
*k*?  It's actually a spread across the entire spectrum: the first few will
most certainly be the largest singular values, and the bottom few will be
the smallest, but you have no guarantee that just because you have the n'th
largest singular value of A, that you also have the (n-1)'st as well.  A
good rule of thumb is to try and extract out the top 3k singular vectors
via Lanczos, and then discard the bottom two thirds, if you want primarily
the largest singular values (which is the case for using Lanczos for
dimensional reduction).

<a name="DimensionalReduction-ParallelizationStragegy"></a>
### Parallelization Stragegy

Lanczos is "embarassingly parallelizable": matrix multiplication of a
matrix by a vector may be carried out row-at-a-time without communication
until at the end, the results of the intermediate matrix-by-vector outputs
are accumulated on one final vector.  When it's truly A.times(*v*), the
final accumulation doesn't even have collision / synchronization issues
(the outputs are individual separate entries on a single vector), and
multicore approaches can be very fast, and there should also be tricks to
speed things up on Hadoop.  In the asymmetric case, where the operation is
A.timesSquared(*v*), the accumulation does require synchronization (the
vectors to be summed have nonzero elements all across their range), but
delaying writing to disk until Mapper close(), and remembering that having
a Combiner be the same as the Reducer, the bottleneck in accumulation is
nowhere near a single point.

<a name="DimensionalReduction-Mahoutusage"></a>
### Mahout usage

The Mahout DistributedLanzcosSolver is invoked by the
<MAHOUT_HOME>/bin/mahout svd command. This command takes the following
arguments (which can be reproduced by just entering the command with no
arguments):


    Job-Specific Options:							    
      --input (-i) input			  Path to job input directory.	    
      --output (-o) output			  The directory pathname for output.    
      --numRows (-nr) numRows		  Number of rows of the input matrix	  
      --numCols (-nc) numCols		  Number of columns of the input matrix 
      --rank (-r) rank			  Desired decomposition rank (note: 
    					  only roughly 1/4 to 1/3 of these will 
    					  have the top portion of the spectrum) 
      --symmetric (-sym) symmetric		  Is the input matrix square and    
    					  symmetric?			    
      --cleansvd (-cl) cleansvd		  Run the EigenVerificationJob to clean 
    					  the eigenvectors after SVD	    
      --maxError (-err) maxError		  Maximum acceptable error	    
      --minEigenvalue (-mev) minEigenvalue	  Minimum eigenvalue to keep the vector for				    
      --inMemory (-mem) inMemory		  Buffer eigen matrix into memory (if you have enough!)		    
      --help (-h)				  Print out help		    
      --tempDir tempDir			  Intermediate output directory     
      --startPhase startPhase		  First phase to run		    
      --endPhase endPhase			  Last phase to run		    


The short form invocation may be used to perform the SVD on the input data: 

      <MAHOUT_HOME>/bin/mahout svd \
      --input (-i) <Path to input matrix> \   
      --output (-o) <The directory pathname for output> \	
      --numRows (-nr) <Number of rows of the input matrix> \   
      --numCols (-nc) <Number of columns of the input matrix> \
      --rank (-r) <Desired decomposition rank> \
      --symmetric (-sym) <Is the input matrix square and symmetric>    


The --input argument is the location on HDFS where a
SequenceFile<Writable,VectorWritable> (preferably
SequentialAccessSparseVectors instances) lies which you wish to decompose. 
Each vector of which has --numcols entries.  --numRows is the number of
input rows and is used to properly size the matrix data structures.

After execution, the --output directory will have a file named
"rawEigenvectors" containing the raw eigenvectors. As the
DistributedLanczosSolver sometimes produces "extra" eigenvectors, whose
eigenvalues aren't valid, and also scales all of the eigenvalues down by
the max eignenvalue (to avoid floating point overflow), there is an
additional step which spits out the nice correctly scaled (and
non-spurious) eigenvector/value pairs. This is done by the "cleansvd" shell
script step (c.f. EigenVerificationJob).

If you have run he short form svd invocation above and require this
"cleaning" of the eigen/singular output you can run "cleansvd" as a
separate command:

      <MAHOUT_HOME>/bin/mahout cleansvd \
      --eigenInput <path to raw eigenvectors> \
      --corpusInput <path to corpus> \
      --output <path to output directory> \
      --maxError <maximum allowed error. Default is 0.5> \
      --minEigenvalue <minimum allowed eigenvalue. Default is 0.0> \
      --inMemory <true if the eigenvectors can all fit into memory. Default false>


The --corpusInput is the input path from the previous step, --eigenInput is
the output from the previous step (<output>/rawEigenvectors), and --output
is the desired output path (same as svd argument). The two "cleaning"
params are --maxError - the maximum allowed 1-cosAngle(v,
A.timesSquared(v)), and --minEigenvalue.  Eigenvectors which have too large
error, or too small eigenvalue are discarded.  Optional argument:
--inMemory, if you have enough memory on your local machine (not on the
hadoop cluster nodes!) to load all eigenvectors into memory at once (at
least 8 bytes/double * rank * numCols), then you will see some speedups on
this cleaning process.

After execution, the --output directory will have a file named
"cleanEigenvectors" containing the clean eigenvectors. 

These two steps can also be invoked together by the svd command by using
the long form svd invocation:

      <MAHOUT_HOME>/bin/mahout svd \
      --input (-i) <Path to input matrix> \   
      --output (-o) <The directory pathname for output> \	
      --numRows (-nr) <Number of rows of the input matrix> \   
      --numCols (-nc) <Number of columns of the input matrix> \
      --rank (-r) <Desired decomposition rank> \
      --symmetric (-sym) <Is the input matrix square and symmetric> \  
      --cleansvd "true"   \
      --maxError <maximum allowed error. Default is 0.5> \
      --minEigenvalue <minimum allowed eigenvalue. Default is 0.0> \
      --inMemory <true if the eigenvectors can all fit into memory. Default false>


After execution, the --output directory will contain two files: the
"rawEigenvectors" and the "cleanEigenvectors".

TODO: also allow exclusion based on improper orthogonality (currently
computed, but not checked against constraints).

<a name="DimensionalReduction-Example:SVDofASFMailArchivesonAmazonElasticMapReduce"></a>
#### Example: SVD of ASF Mail Archives on Amazon Elastic MapReduce

This section walks you through a complete example of running the Mahout SVD
job on Amazon Elastic MapReduce cluster and then preparing the output to be
used for clustering. This example was developed as part of the effort to
benchmark Mahout's clustering algorithms using a large document set (see [MAHOUT-588](https://issues.apache.org/jira/browse/MAHOUT-588)
). Specifically, we use the ASF mail archives located at
http://aws.amazon.com/datasets/7791434387204566.  You will need to likely
run seq2sparse on these first.	See
$MAHOUT_HOME/examples/bin/build-asf-email.sh (on trunk) for examples of
processing this data.

At a high level, the steps we're going to perform are:

bin/mahout svd (original -> svdOut)
bin/mahout cleansvd ...
bin/mahout transpose svdOut -> svdT
bin/mahout transpose original -> originalT
bin/mahout matrixmult originalT svdT -> newMatrix
bin/mahout kmeans newMatrix

The bulk of the content for this section was extracted from the Mahout user
mailing list, see: [Using SVD with Canopy/KMeans](http://search.lucidimagination.com/search/document/6e5889ee6f0f253b/using_svd_with_canopy_kmeans#66a50fe017cebbe8)
 and [Need a little help with using SVD](http://search.lucidimagination.com/search/document/748181681ae5238b/need_a_little_help_with_using_svd#134fb2771fd52928)

Note: Some of this work is due in part to credits donated by the Amazon
Elastic MapReduce team.

<a name="DimensionalReduction-1.LaunchEMRCluster"></a>
##### 1. Launch EMR Cluster

For a detailed explanation of the steps involved in launching an Amazon
Elastic MapReduce cluster for running Mahout jobs, please read the
"Building Vectors for Large Document Sets" section of [Mahout on Elastic MapReduce](https://cwiki.apache.org/confluence/display/MAHOUT/Mahout+on+Elastic+MapReduce)
.

In the remaining steps below, remember to replace JOB_ID with the Job ID of
your EMR cluster.

<a name="DimensionalReduction-2.LoadMahout0.5+JARintoS3"></a>
##### 2. Load Mahout 0.5+ JAR into S3

These steps were created with the mahout-0.5-SNAPSHOT because they rely on
the patch for [MAHOUT-639](https://issues.apache.org/jira/browse/MAHOUT-639)

<a name="DimensionalReduction-3.CopyTFIDFVectorsintoHDFS"></a>
##### 3. Copy TFIDF Vectors into HDFS

Before running your SVD job on the vectors, you need to copy them from S3
to your EMR cluster's HDFS.


    elastic-mapreduce --jar s3://elasticmapreduce/samples/distcp/distcp.jar \
      --arg s3n://ACCESS_KEY:SECRET_KEY@asf-mail-archives/mahout-0.4/sparse-1-gram-stem/tfidf-vectors\
      --arg /asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-vectors \
      -j JOB_ID


<a name="DimensionalReduction-4.RuntheSVDJob"></a>
##### 4. Run the SVD Job

Now you're ready to run the SVD job on the vectors stored in HDFS:


    elastic-mapreduce --jar s3://BUCKET/mahout-examples-0.5-SNAPSHOT-job.jar \
      --main-class org.apache.mahout.driver.MahoutDriver \
      --arg svd \
      --arg -i --arg /asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-vectors\
      --arg -o --arg /asf-mail-archives/mahout/svd \
      --arg --rank --arg 100 \
      --arg --numCols --arg 20444 \
      --arg --numRows --arg 6076937 \
      --arg --cleansvd --arg "true" \
      -j JOB_ID


This will run 100 iterations of the LanczosSolver SVD job to produce 87
eigenvectors in:


    /asf-mail-archives/mahout/svd/cleanEigenvectors


Only 87 eigenvectors were produced because of the cleanup step, which
removes any duplicate eigenvectors caused by convergence issues and numeric
overflow and any that don't appear to be "eigen" enough (ie, they don't
satisfy the eigenvector criterion with high enough fidelity). - Jake Mannix

<a name="DimensionalReduction-5.TransformyourTFIDFVectorsintoMahoutMatrix"></a>
##### 5. Transform your TFIDF Vectors into Mahout Matrix

The tfidf vectors created by the seq2sparse job are
SequenceFile<Text,VectorWritable>. The Mahout RowId job transforms these
vectors into a matrix form that is a
SequenceFile<IntWritable,VectorWritable> and a
SequenceFile<IntWritable,Text> (where the original one is the join of these
new ones, on the new int key).


    elastic-mapreduce --jar s3://BUCKET/mahout-examples-0.5-SNAPSHOT-job.jar \
      --main-class org.apache.mahout.driver.MahoutDriver \
      --arg rowid \
      --arg
-Dmapred.input.dir=/asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-vectors
\
      --arg
-Dmapred.output.dir=/asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-matrix
\
      -j JOB_ID


This is not a distributed job and will only run on the master server in
your EMR cluster. The job produces the following output:


    /asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-matrix/docIndex
    /asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-matrix/matrix


where docIndex is the SequenceFile<IntWritable,Text> and matrix is
SequenceFile<IntWritable,VectorWritable>.

<a name="DimensionalReduction-6.TransposetheMatrix"></a>
##### 6. Transpose the Matrix

Our ultimate goal is to multiply the TFIDF vector matrix times our SVD
eigenvectors. For the mathematically inclined, from the rowid job, we now
have an m x n matrix T (m=6076937, n=20444). The SVD eigenvector matrix E
is p x n (p=87, n=20444). So to multiply these two matrices, I need to
transpose E so that the number of columns in T equals the number of rows in
E (i.e. E^T is n x p) the result of the matrixmult would give me an m x p
matrix (m=6076937, p=87).

However, in practice, computing the matrix product of two matrices as a
map-reduce job is efficiently done as a map-side join on two row-based
matrices with the same number of rows, and the columns are the ones which
are different.	In particular, if you take a matrix X which is represented
as a set of numRowsX rows, each of which has numColsX, and another matrix
with numRowsY == numRowsX, each of which has numColsY (!= numColsX), then
by summing the outer-products of each of the numRowsX pairs of vectors, you
get a matrix of with numRowsZ == numColsX, and numColsZ == numColsY (if you
instead take the reverse outer product of the vector pairs, you can end up
with the transpose of this final result, with numRowsZ == numColsY, and
numColsZ == numColsX). - Jake Mannix

Thus, we need to transpose the matrix using Mahout's Transpose Job:


    elastic-mapreduce --jar s3://BUCKET/mahout-examples-0.5-SNAPSHOT-job.jar \
      --main-class org.apache.mahout.driver.MahoutDriver \
      --arg transpose \
      --arg -i --arg
/asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-matrix/matrix \
      --arg --numRows --arg 6076937 \
      --arg --numCols --arg 20444 \
      --arg --tempDir --arg
/asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-matrix/transpose \
      -j JOB_ID


This job requires the patch to [MAHOUT-639](https://issues.apache.org/jira/browse/MAHOUT-639)

The job creates the following output:


    /asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-matrix/transpose


<a name="DimensionalReduction-7.TransposeEigenvectors"></a>
##### 7. Transpose Eigenvectors

If you followed Jake's explanation in step 6 above, then you know that we
also need to transpose the eigenvectors:


    elastic-mapreduce --jar s3://BUCKET/mahout-examples-0.5-SNAPSHOT-job.jar \
      --main-class org.apache.mahout.driver.MahoutDriver \
      --arg transpose \
      --arg -i --arg /asf-mail-archives/mahout/svd/cleanEigenvectors \
      --arg --numRows --arg 87 \
      --arg --numCols --arg 20444 \
      --arg --tempDir --arg /asf-mail-archives/mahout/svd/transpose \
      -j JOB_ID


Note: You need to use the same number of reducers that was used for
transposing the matrix you are multiplying the vectors with.

The job creates the following output:


    /asf-mail-archives/mahout/svd/transpose


<a name="DimensionalReduction-8.MatrixMultiplication"></a>
##### 8. Matrix Multiplication

Lastly, we need to multiply the transposed vectors using Mahout's
matrixmult job:


    elastic-mapreduce --jar s3://BUCKET/mahout-examples-0.5-SNAPSHOT-job.jar \
      --main-class org.apache.mahout.driver.MahoutDriver \
      --arg matrixmult \
      --arg --numRowsA --arg 20444 \
      --arg --numColsA --arg 6076937 \
      --arg --numRowsB --arg 20444 \
      --arg --numColsB --arg 87 \
      --arg --inputPathA --arg
/asf-mail-archives/mahout/sparse-1-gram-stem/tfidf-matrix/transpose \
      --arg --inputPathB --arg /asf-mail-archives/mahout/svd/transpose \
      -j JOB_ID


This job produces output such as:


    /user/hadoop/productWith-189


<a name="DimensionalReduction-Resources"></a>
# Resources

* [LSA tutorial](http://www.dcs.shef.ac.uk/~genevieve/lsa_tutorial.htm)
* [SVD tutorial](http://www.puffinwarellc.com/index.php/news-and-articles/articles/30-singular-value-decomposition-tutorial.html)
