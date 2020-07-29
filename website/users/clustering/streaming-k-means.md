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
title: Spectral Clustering

   
---

# *StreamingKMeans* algorithm 

The *StreamingKMeans* algorithm is a variant of Algorithm 1 from [Shindler et al][1] and consists of two steps:

 1. Streaming step 
 2. BallKMeans step. 

The streaming step is a randomized algorithm that makes one pass through the data and 
produces as many centroids as it determines is optimal. This step can be viewed as 
a preparatory dimensionality reduction. If the size of the data stream is *n* and the 
expected number of clusters is *k*, the streaming step will produce roughly *k\*log(n)* 
clusters that will be passed on to the BallKMeans step which will further reduce the 
number of clusters down to *k*. BallKMeans is a randomized Lloyd-type algorithm that
has been studied in detail, see [Ostrovsky et al][2].

## Streaming step

---

### Overview

The streaming step is a derivative of the streaming 
portion of Algorithm 1 in [Shindler et al][1]. The main difference between the two is that 
Algorithm 1 of [Shindler et al][1] assumes 
the knowledge of the size of the data stream and uses it to set a key parameter 
for the algorithm. More precisely, the initial *distanceCutoff* (defined below), which is 
denoted by *f* in [Shindler et al][1], is set to *1/(k(1+log(n))*. The *distanceCutoff* influences the number of clusters that the algorithm 
will produce. 
In contrast, Mahout implementation does not require the knowledge of the size of the 
data stream. Instead, it dynamically re-evaluates the parameters that depend on the size 
of the data stream at runtime as more and more data is processed. In particular, 
the parameter *numClusters* (defined below) changes its value as the data is processed.   

###Parameters

 - **numClusters** (int): Conceptually, *numClusters* represents the algorithm's guess at the optimal 
number of clusters it is shooting for. In particular, *numClusters* will increase at run 
time as more and more data is processed. Note that •numClusters• is not the number of clusters that the algorithm will produce. Also, *numClusters* should not be set to the final number of clusters that we expect to receive as the output of *StreamingKMeans*. 
 - **distanceCutoff** (double): a parameter representing the value of the distance between a point and 
its closest centroid after which
the new point will definitely be assigned to a new cluster. *distanceCutoff* can be thought 
of as an estimate of the variable *f* from Shindler et al. The default initial value for 
*distanceCutoff* is *1.0/numClusters* and *distanceCutoff* grows as a geometric progression with 
common ratio *beta* (see below).    
 - **beta** (double): a constant parameter that controls the growth of *distanceCutoff*. If the initial setting of *distanceCutoff* is *d0*, *distanceCutoff* will grow as the geometric progression with initial term *d0* and common ratio *beta*. The default value for *beta* is 1.3. 
 - **clusterLogFactor** (double): a constant parameter such that *clusterLogFactor* *log(numProcessedPoints)* is the runtime estimate of the number of clusters to be produced by the streaming step. If the final number of clusters (that we expect *StreamingKMeans* to output) is *k*, *clusterLogFactor* can be set to *k*.  
 - **clusterOvershoot** (double): a constant multiplicative slack factor that slows down the collapsing of clusters. The default value is 2. 


###Algorithm 

The algorithm processes the data one-by-one and makes only one pass through the data.
The first point from the data stream will form the centroid of the first cluster (this designation may change as more points are processed). Suppose there are *r* clusters at one point and a new point *p* is being processed. The new point can either be added to one of the existing *r* clusters or become a new cluster. To decide:

 - let *c* be the closest cluster to point *p*
 - let *d* be the distance between *c* and *p*
 - if *d > distanceCutoff*, create a new cluster from *p* (*p* is too far away from the clusters to be part of any one of them)
 - else (*d <= distanceCutoff*), create a new cluster with probability *d / distanceCutoff* (the probability of creating a new cluster increases as *d* increases). 

There will be either *r* or *r+1* clusters after processing a new point.

As the number of clusters increases, it will go over the  *clusterOvershoot \* numClusters* limit (*numClusters* represents a recommendation for the number of clusters that the streaming step should aim for and *clusterOvershoot* is the slack). To decrease the number of clusters the existing clusters
are treated as data points and are re-clustered (collapsed). This tends to make the number of clusters go down. If the number of clusters is still too high, *distanceCutoff* is increased.

## BallKMeans step
---
### Overview
The algorithm is a Lloyd-type algorithm that takes a set of weighted vectors and returns k centroids, see [Ostrovsky et al][2] for details. The algorithm has two stages: 
 
 1. Seeding 
 2. Ball k-means 

The seeding stage is an initial guess of where the centroids should be. The initial guess is improved using the ball k-means stage. 

### Parameters

* **numClusters** (int): the number k of centroids to return.  The algorithm will return exactly this number of centroids.

* **maxNumIterations** (int): After seeding, the iterative clustering procedure will be run at most *maxNumIterations* times.  1 or 2 iterations are recommended.  Increasing beyond this will increase the accuracy of the result at the expense of runtime.  Each successive iteration yields diminishing returns in lowering the cost.

* **trimFraction** (double): Outliers are ignored when computing the center of mass for a cluster.  For any datapoint *x*, let *c* be the nearest centroid.  Let *d* be the minimum distance from *c* to another centroid.  If the distance from *x* to *c* is greater than *trimFraction \* d*, then *x* is considered an outlier during that iteration of ball k-means.  The default is 9/10.  In [Ostrovsky et al][2], the authors use *trimFraction* = 1/3, but this does not mean that 1/3 is optimal in practice.

* **kMeansPlusPlusInit** (boolean): If true, the seeding method is k-means++.  If false, the seeding method is to select points uniformly at random.  The default is true.

* **correctWeights** (boolean): If *correctWeights* is true, outliers will be considered when calculating the weight of centroids.  The default is true. Note that outliers are not considered when calculating the position of centroids.

* **testProbability** (double): If *testProbability* is *p* (0 < *p* < 1), the data (of size n) is partitioned into a test set (of size *p\*n*) and a training set (of size *(1-p)\*n*).  If 0, no test set is created (the entire data set is used for both training and testing).  The default is 0.1 if *numRuns* > 1.  If *numRuns* = 1, then no test set should be created (since it is only used to compare the cost between different runs).

* **numRuns** (int): This is the number of runs to perform. The solution of lowest cost is returned.  The default is 1 run.

###Algorithm
The algorithm can be instructed to take multiple independent runs (using the *numRuns* parameter) and the algorithm will select the best solution (i.e., the one with the lowest cost). In practice, one run is sufficient to find a good solution.  

Each run operates as follows: a seeding procedure is used to select k centroids, and then ball k-means is run iteratively to refine the solution.

The seeding procedure can be set to either 'uniformly at random' or 'k-means++' using *kMeansPlusPlusInit* boolean variable. Seeding with k-means++ involves more computation but offers better results in practice. 
 
Each iteration of ball k-means runs as follows:

1. Clusters are formed by assigning each datapoint to the nearest centroid
2. The centers of mass of the trimmed clusters (see *trimFraction* parameter above) become the new centroids 

The data may be partitioned into a test set and a training set (see *testProbability*). The seeding procedure and ball k-means run on the training set. The cost is computed on the test set.


##Usage of *StreamingKMeans*                                                                          
  
     bin/mahout streamingkmeans  
       -i <input>  
       -o <output> 
       -ow  
       -k <k>  
       -km <estimatedNumMapClusters>  
       -e <estimatedDistanceCutoff>  
       -mi <maxNumIterations>  
       -tf <trimFraction>  
       -ri                  
       -iw  
       -testp <testProbability>  
       -nbkm <numBallKMeansRuns>  
       -dm <distanceMeasure>   
       -sc <searcherClass>  
       -np <numProjections>  
       -s <searchSize>   
       -rskm  
       -xm <method>  
       -h   
       --tempDir <tempDir>   
       --startPhase <startPhase>   
       --endPhase <endPhase>                    


###Details on Job-Specific Options:
                                                           
 * `--input (-i) <input>`: Path to job input directory.         
 * `--output (-o) <output>`: The directory pathname for output.            
 * `--overwrite (-ow)`: If present, overwrite the output directory before running job.
 * `--numClusters (-k) <k>`: The k in k-Means. Approximately this many clusters will be generated.      
 * `--estimatedNumMapClusters (-km) <estimatedNumMapClusters>`: The estimated number of clusters to use for the Map phase of the job when running StreamingKMeans. This should be around k \* log(n), where k is the final number of clusters and n is the total number of data points to cluster.           
 * `--estimatedDistanceCutoff (-e) <estimatedDistanceCutoff>`: The initial estimated distance cutoff between two points for forming new clusters. If no value is given, it's estimated from the data set  
 * `--maxNumIterations (-mi) <maxNumIterations>`: The maximum number of iterations to run for the BallKMeans algorithm used by the reducer. If no value is given, defaults to 10.    
 * `--trimFraction (-tf) <trimFraction>`: The 'ball' aspect of ball k-means means that only the closest points to the centroid will actually be used for updating. The fraction of the points to be used is those points whose distance to the center is within trimFraction \* distance to the closest other center. If no value is given, defaults to 0.9.   
 * `--randomInit` (`-ri`) Whether to use k-means++ initialization or random initialization of the seed centroids. Essentially, k-means++ provides better clusters, but takes longer, whereas random initialization takes less time, but produces worse clusters, and tends to fail more often and needs multiple runs to compare to k-means++. If set, uses the random initialization.
 * `--ignoreWeights (-iw)`: Whether to correct the weights of the centroids after the clustering is done. The weights end up being wrong because of the trimFraction and possible train/test splits. In some cases, especially in a pipeline, having an accurate count of the weights is useful. If set, ignores the final weights. 
 * `--testProbability (-testp) <testProbability>`: A double value  between 0 and 1  that represents  the percentage of  points to be used  for 'testing'  different  clustering runs in  the final  BallKMeans step.  If no value is  given, defaults to  0.1  
 * `--numBallKMeansRuns (-nbkm) <numBallKMeansRuns>`: Number of  BallKMeans runs to  use at the end to  try to cluster the  points. If no  value is given,  defaults to 4  
 * `--distanceMeasure (-dm) <distanceMeasure>`: The classname of  the  DistanceMeasure.  Default is  SquaredEuclidean.  
 * `--searcherClass (-sc) <searcherClass>`: The type of  searcher to be  used when  performing nearest  neighbor searches.  Defaults to  ProjectionSearch.  
 * `--numProjections (-np) <numProjections>`: The number of  projections  considered in  estimating the  distances between  vectors. Only used  when the distance  measure requested is either ProjectionSearch or FastProjectionSearch. If no value is given, defaults to 3.  
 * `--searchSize (-s) <searchSize>`: In more efficient  searches (non  BruteSearch), not all distances are calculated for determining the nearest neighbors. The number of elements whose distances from the query vector is actually computer is proportional to searchSize. If no value is given, defaults to 1.  
 * `--reduceStreamingKMeans (-rskm)`: There might be too many intermediate clusters from the mapper to fit into memory, so the reducer can run  another pass of StreamingKMeans to collapse them down to a fewer clusters.  
 * `--method (-xm)` method The execution  method to use:  sequential or  mapreduce. Default  is mapreduce.  
 * `-- help (-h)`: Print out help  
 * `--tempDir <tempDir>`: Intermediate output directory.
 * `--startPhase <startPhase>` First phase to run.  
 * `--endPhase <endPhase>` Last phase to run.   


##References

1. [M. Shindler, A. Wong, A. Meyerson: Fast and Accurate k-means For Large Datasets][1]
2. [R. Ostrovsky, Y. Rabani, L. Schulman, Ch. Swamy: The Effectiveness of Lloyd-Type Methods for the k-means Problem][2]


[1]: http://nips.cc/Conferences/2011/Program/event.php?ID=2989 "M. Shindler, A. Wong, A. Meyerson: Fast and Accurate k-means For Large Datasets"

[2]: http://www.math.uwaterloo.ca/~cswamy/papers/kmeansfnl.pdf "R. Ostrovsky, Y. Rabani, L. Schulman, Ch. Swamy: The Effectiveness of Lloyd-Type Methods for the k-means Problem"
