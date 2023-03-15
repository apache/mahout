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
title: ClusteringYourData

   
---

# Clustering your data

After you've done the [Quickstart](quickstart.html) and are familiar with the basics of Mahout, it is time to cluster your own
data. See also [Wikipedia on cluster analysis](en.wikipedia.org/wiki/Cluster_analysis) for more background.

The following pieces *may* be useful for in getting started:

<a name="ClusteringYourData-Input"></a>
# Input

For starters, you will need your data in an appropriate Vector format, see [Creating Vectors](../basics/creating-vectors.html).
In particular for text preparation check out [Creating Vectors from Text](../basics/creating-vectors-from-text.html).


<a name="ClusteringYourData-RunningtheProcess"></a>
# Running the Process

* [Canopy background](canopy-clustering.html) and [canopy-commandline](canopy-commandline.html).

* [K-Means background](k-means-clustering.html), [k-means-commandline](k-means-commandline.html), and
[fuzzy-k-means-commandline](fuzzy-k-means-commandline.html).

* [Dirichlet background](dirichlet-process-clustering.html) and [dirichlet-commandline](dirichlet-commandline.html).

* [Meanshift background](mean-shift-clustering.html) and [mean-shift-commandline](mean-shift-commandline.html).

* [LDA (Latent Dirichlet Allocation) background](-latent-dirichlet-allocation.html) and [lda-commandline](lda-commandline.html).

* TODO: kmeans++/ streaming kMeans documentation


<a name="ClusteringYourData-RetrievingtheOutput"></a>
# Retrieving the Output

Mahout has a cluster dumper utility that can be used to retrieve and evaluate your clustering data.

    ./bin/mahout clusterdump <OPTIONS>


<a name="ClusteringYourData-Theclusterdumperoptionsare:"></a>
## The cluster dumper options are:

      --help (-h)				   Print out help	
	    
      --input (-i) input			   The directory containing Sequence    
    					   Files for the Clusters	    

      --output (-o) output			   The output file.  If not specified,  
    					   dumps to the console.

      --outputFormat (-of) outputFormat	   The optional output format to write
    					   the results as. Options: TEXT, CSV, or GRAPH_ML		 

      --substring (-b) substring		   The number of chars of the	    
    					   asFormatString() to print	
    
      --pointsDir (-p) pointsDir		   The directory containing points  
 					   sequence files mapping input vectors     					   to their cluster.  If specified, 
    					   then the program will output the 
    					   points associated with a cluster 

      --dictionary (-d) dictionary		   The dictionary file. 	    

      --dictionaryType (-dt) dictionaryType    The dictionary file type	    
    					   (text|sequencefile)

      --distanceMeasure (-dm) distanceMeasure  The classname of the DistanceMeasure.
    					   Default is SquaredEuclidean.     

      --numWords (-n) numWords		   The number of top terms to print 

      --tempDir tempDir			   Intermediate output directory

      --startPhase startPhase		   First phase to run

      --endPhase endPhase			   Last phase to run

      --evaluate (-e)			   Run ClusterEvaluator and CDbwEvaluator over the
    					   input. The output will be appended to the rest of
    					   the output at the end.   


More information on using clusterdump utility can be found [here](cluster-dumper.html)

<a name="ClusteringYourData-ValidatingtheOutput"></a>
# Validating the Output

{quote}
Ted Dunning: A principled approach to cluster evaluation is to measure how well the
cluster membership captures the structure of unseen data.  A natural
measure for this is to measure how much of the entropy of the data is
captured by cluster membership.  For k-means and its natural L_2 metric,
the natural cluster quality metric is the squared distance from the nearest
centroid adjusted by the log_2 of the number of clusters.  This can be
compared to the squared magnitude of the original data or the squared
deviation from the centroid for all of the data.  The idea is that you are
changing the representation of the data by allocating some of the bits in
your original representation to represent which cluster each point is in. 
If those bits aren't made up by the residue being small then your
clustering is making a bad trade-off.

In the past, I have used other more heuristic measures as well.  One of the
key characteristics that I would like to see out of a clustering is a
degree of stability.  Thus, I look at the fractions of points that are
assigned to each cluster or the distribution of distances from the cluster
centroid. These values should be relatively stable when applied to held-out
data.

For text, you can actually compute perplexity which measures how well
cluster membership predicts what words are used.  This is nice because you
don't have to worry about the entropy of real valued numbers.

Manual inspection and the so-called laugh test is also important.  The idea
is that the results should not be so ludicrous as to make you laugh.
Unfortunately, it is pretty easy to kid yourself into thinking your system
is working using this kind of inspection.  The problem is that we are too
good at seeing (making up) patterns.
{quote}

