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
title: k-means-commandline

   
---

<a name="k-means-commandline-Introduction"></a>
# kMeans commandline introduction

This quick start page describes how to run the kMeans clustering algorithm
on a Hadoop cluster. 

<a name="k-means-commandline-Steps"></a>
# Steps

Mahout's k-Means clustering can be launched from the same command line
invocation whether you are running on a single machine in stand-alone mode
or on a larger Hadoop cluster. The difference is determined by the
$HADOOP_HOME and $HADOOP_CONF_DIR environment variables. If both are set to
an operating Hadoop cluster on the target machine then the invocation will
run k-Means on that cluster. If either of the environment variables are
missing then the stand-alone Hadoop configuration will be invoked instead.


    ./bin/mahout kmeans <OPTIONS>


In $MAHOUT_HOME/, build the jar containing the job (mvn install) The job
will be generated in $MAHOUT_HOME/core/target/ and it's name will contain
the Mahout version number. For example, when using Mahout 0.3 release, the
job will be mahout-core-0.3.job


<a name="k-means-commandline-Testingitononesinglemachinew/ocluster"></a>
## Testing it on one single machine w/o cluster

* Put the data: cp <PATH TO DATA> testdata
* Run the Job: 

    ./bin/mahout kmeans -i testdata -o output -c clusters -dm
org.apache.mahout.common.distance.CosineDistanceMeasure -x 5 -ow -cd 1 -k
25


<a name="k-means-commandline-Runningitonthecluster"></a>
## Running it on the cluster

* (As needed) Start up Hadoop: $HADOOP_HOME/bin/start-all.sh
* Put the data: $HADOOP_HOME/bin/hadoop fs -put <PATH TO DATA> testdata
* Run the Job: 

    export HADOOP_HOME=<Hadoop Home Directory>
    export HADOOP_CONF_DIR=$HADOOP_HOME/conf
    ./bin/mahout kmeans -i testdata -o output -c clusters -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 5 -ow -cd 1 -k 25

* Get the data out of HDFS and have a look. Use bin/hadoop fs -lsr output
to view all outputs.

<a name="k-means-commandline-Commandlineoptions"></a>
# Command line options

      --input (-i) input			       Path to job input directory. 
    					       Must be a SequenceFile of    
    					       VectorWritable		    
      --clusters (-c) clusters		       The input centroids, as Vectors. 
    					       Must be a SequenceFile of    
    					       Writable, Cluster/Canopy. If k  
    					       is also specified, then a random 
    					       set of vectors will be selected  
    					       and written out to this path 
    					       first			    
      --output (-o) output			       The directory pathname for   
    					       output.			    
      --distanceMeasure (-dm) distanceMeasure      The classname of the	    
    					       DistanceMeasure. Default is  
    					       SquaredEuclidean 	    
      --convergenceDelta (-cd) convergenceDelta    The convergence delta value. 
    					       Default is 0.5		    
      --maxIter (-x) maxIter		       The maximum number of	    
    					       iterations.		    
      --maxRed (-r) maxRed			       The number of reduce tasks.  
    					       Defaults to 2		    
      --k (-k) k				       The k in k-Means.  If specified, 
    					       then a random selection of k 
    					       Vectors will be chosen as the    
    					       Centroid and written to the  
    					       clusters input path.	    
      --overwrite (-ow)			       If present, overwrite the output 
    					       directory before running job 
      --help (-h)				       Print out help		    
      --clustering (-cl)			       If present, run clustering after 
    					       the iterations have taken place  

