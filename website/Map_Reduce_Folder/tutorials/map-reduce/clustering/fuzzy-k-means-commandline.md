---
layout: deprecated-docs
title: (Deprecated)  fuzzy-k-means-commandline
redirect_from:
	- /documentation/tutorials/map-reduce/clustering/fuzzy-k-means-commandline
	- /documentation/tutorials/map-reduce/clustering/fuzzy-k-means-commandline.html
---



<a name="fuzzy-k-means-commandline-RunningFuzzyk-MeansClusteringfromtheCommandLine"></a>
# Running Fuzzy k-Means Clustering from the Command Line
Mahout's Fuzzy k-Means clustering can be launched from the same command
line invocation whether you are running on a single machine in stand-alone
mode or on a larger Hadoop cluster. The difference is determined by the
$HADOOP_HOME and $HADOOP_CONF_DIR environment variables. If both are set to
an operating Hadoop cluster on the target machine then the invocation will
run FuzzyK on that cluster. If either of the environment variables are
missing then the stand-alone Hadoop configuration will be invoked instead.


    ./bin/mahout fkmeans <OPTIONS>


* In $MAHOUT_HOME/, build the jar containing the job (mvn install) The job
will be generated in $MAHOUT_HOME/core/target/ and it's name will contain
the Mahout version number. For example, when using Mahout 0.3 release, the
job will be mahout-core-0.3.job


<a name="fuzzy-k-means-commandline-Testingitononesinglemachinew/ocluster"></a>
## Testing it on one single machine w/o cluster

* Put the data: cp <PATH TO DATA> testdata
* Run the Job: 

    ./bin/mahout fkmeans -i testdata <OPTIONS>


<a name="fuzzy-k-means-commandline-Runningitonthecluster"></a>
## Running it on the cluster

* (As needed) Start up Hadoop: $HADOOP_HOME/bin/start-all.sh
* Put the data: $HADOOP_HOME/bin/hadoop fs -put <PATH TO DATA> testdata
* Run the Job: 

    export HADOOP_HOME=<Hadoop Home Directory>
    export HADOOP_CONF_DIR=$HADOOP_HOME/conf
    ./bin/mahout fkmeans -i testdata <OPTIONS>

* Get the data out of HDFS and have a look. Use bin/hadoop fs -lsr output
to view all outputs.

<a name="fuzzy-k-means-commandline-Commandlineoptions"></a>
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
      --k (-k) k				       The k in k-Means.  If specified, 
    					       then a random selection of k 
    					       Vectors will be chosen as the
        					       Centroid and written to the  
    					       clusters input path.	    
      --m (-m) m				       coefficient normalization    
    					       factor, must be greater than 1   
      --overwrite (-ow)			       If present, overwrite the output 
    					       directory before running job 
      --help (-h)				       Print out help		    
      --numMap (-u) numMap			       The number of map tasks.     
    					       Defaults to 10		    
      --maxRed (-r) maxRed			       The number of reduce tasks.  
    					       Defaults to 2		    
      --emitMostLikely (-e) emitMostLikely	       True if clustering should emit   
    					       the most likely point only,  
    					       false for threshold clustering.  
    					       Default is true		    
      --threshold (-t) threshold		       The pdf threshold used for   
    					       cluster determination. Default   
    					       is 0 
      --clustering (-cl)			       If present, run clustering after 
    					       the iterations have taken place  
                                

