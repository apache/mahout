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
title: (Deprecated)  canopy-commandline

   
---

<a name="canopy-commandline-RunningCanopyClusteringfromtheCommandLine"></a>
# Running Canopy Clustering from the Command Line
Mahout's Canopy clustering can be launched from the same command line
invocation whether you are running on a single machine in stand-alone mode
or on a larger Hadoop cluster. The difference is determined by the
$HADOOP_HOME and $HADOOP_CONF_DIR environment variables. If both are set to
an operating Hadoop cluster on the target machine then the invocation will
run Canopy on that cluster. If either of the environment variables are
missing then the stand-alone Hadoop configuration will be invoked instead.


    ./bin/mahout canopy <OPTIONS>


* In $MAHOUT_HOME/, build the jar containing the job (mvn install) The job
will be generated in $MAHOUT_HOME/core/target/ and it's name will contain
the Mahout version number. For example, when using Mahout 0.3 release, the
job will be mahout-core-0.3.job


<a name="canopy-commandline-Testingitononesinglemachinew/ocluster"></a>
## Testing it on one single machine w/o cluster

* Put the data: cp <PATH TO DATA> testdata
* Run the Job: 

    ./bin/mahout canopy -i testdata -o output -dm
org.apache.mahout.common.distance.CosineDistanceMeasure -ow -t1 5 -t2 2


<a name="canopy-commandline-Runningitonthecluster"></a>
## Running it on the cluster

* (As needed) Start up Hadoop: $HADOOP_HOME/bin/start-all.sh
* Put the data: $HADOOP_HOME/bin/hadoop fs -put <PATH TO DATA> testdata
* Run the Job: 

    export HADOOP_HOME=<Hadoop Home Directory>
    export HADOOP_CONF_DIR=$HADOOP_HOME/conf
    ./bin/mahout canopy -i testdata -o output -dm
org.apache.mahout.common.distance.CosineDistanceMeasure -ow -t1 5 -t2 2

* Get the data out of HDFS and have a look. Use bin/hadoop fs -lsr output
to view all outputs.

<a name="canopy-commandline-Commandlineoptions"></a>
# Command line options

      --input (-i) input			     Path to job input directory.Must  
    					     be a SequenceFile of	    
    					     VectorWritable		    
      --output (-o) output			     The directory pathname for output. 
      --overwrite (-ow)			     If present, overwrite the output	 
    					     directory before running job   
      --distanceMeasure (-dm) distanceMeasure    The classname of the	    
    					     DistanceMeasure. Default is    
    					     SquaredEuclidean		    
      --t1 (-t1) t1 			     T1 threshold value 	    
      --t2 (-t2) t2 			     T2 threshold value 	    
      --clustering (-cl)			     If present, run clustering after	
    					     the iterations have taken place	 
      --help (-h)				     Print out help		    

