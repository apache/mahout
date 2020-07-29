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
title: lda-commandline

   
---

<a name="lda-commandline-RunningLatentDirichletAllocation(algorithm)fromtheCommandLine"></a>
# Running Latent Dirichlet Allocation (algorithm) from the Command Line
[Since Mahout v0.6](https://issues.apache.org/jira/browse/MAHOUT-897)
 lda has been implemented as Collapsed Variable Bayes (cvb). 

Mahout's LDA can be launched from the same command line invocation whether
you are running on a single machine in stand-alone mode or on a larger
Hadoop cluster. The difference is determined by the $HADOOP_HOME and
$HADOOP_CONF_DIR environment variables. If both are set to an operating
Hadoop cluster on the target machine then the invocation will run the LDA
algorithm on that cluster. If either of the environment variables are
missing then the stand-alone Hadoop configuration will be invoked instead.



    ./bin/mahout cvb <OPTIONS>


* In $MAHOUT_HOME/, build the jar containing the job (mvn install) The job
will be generated in $MAHOUT_HOME/core/target/ and it's name will contain
the Mahout version number. For example, when using Mahout 0.3 release, the
job will be mahout-core-0.3.job


<a name="lda-commandline-Testingitononesinglemachinew/ocluster"></a>
## Testing it on one single machine w/o cluster

* Put the data: cp <PATH TO DATA> testdata
* Run the Job: 

    ./bin/mahout cvb -i testdata <OTHER OPTIONS>


<a name="lda-commandline-Runningitonthecluster"></a>
## Running it on the cluster

* (As needed) Start up Hadoop: $HADOOP_HOME/bin/start-all.sh
* Put the data: $HADOOP_HOME/bin/hadoop fs -put <PATH TO DATA> testdata
* Run the Job: 

    export HADOOP_HOME=<Hadoop Home Directory>
    export HADOOP_CONF_DIR=$HADOOP_HOME/conf
    ./bin/mahout cvb -i testdata <OTHER OPTIONS>

* Get the data out of HDFS and have a look. Use bin/hadoop fs -lsr output
to view all outputs.

<a name="lda-commandline-CommandlineoptionsfromMahoutcvbversion0.8"></a>
# Command line options from Mahout cvb version 0.8

    mahout cvb -h 
      --input (-i) input					  Path to job input directory.	      
      --output (-o) output					  The directory pathname for output.  
      --maxIter (-x) maxIter				  The maximum number of iterations.		
      --convergenceDelta (-cd) convergenceDelta		  The convergence delta value		    
      --overwrite (-ow)					  If present, overwrite the output directory before running job    
      --num_topics (-k) num_topics				  Number of topics to learn		 
      --num_terms (-nt) num_terms				  Vocabulary size   
      --doc_topic_smoothing (-a) doc_topic_smoothing	  Smoothing for document/topic distribution	     
      --term_topic_smoothing (-e) term_topic_smoothing	  Smoothing for topic/term distribution 	 
      --dictionary (-dict) dictionary			  Path to term-dictionary file(s) (glob expression supported) 
      --doc_topic_output (-dt) doc_topic_output		  Output path for the training doc/topic distribution	     
      --topic_model_temp_dir (-mt) topic_model_temp_dir	  Path to intermediate model path (useful for restarting)       
      --iteration_block_size (-block) iteration_block_size	  Number of iterations per perplexity check  
      --random_seed (-seed) random_seed			  Random seed	    
      --test_set_fraction (-tf) test_set_fraction		  Fraction of data to hold out for testing  
      --num_train_threads (-ntt) num_train_threads		  number of threads per mapper to train with  
      --num_update_threads (-nut) num_update_threads	  number of threads per mapper to update the model with	       
      --max_doc_topic_iters (-mipd) max_doc_topic_iters	  max number of iterations per doc for p(topic|doc) learning		  
      --num_reduce_tasks num_reduce_tasks			  number of reducers to use during model estimation 	   
      --backfill_perplexity 				  enable backfilling of missing perplexity values		
      --help (-h)						  Print out help    
      --tempDir tempDir					  Intermediate output directory	     
      --startPhase startPhase				  First phase to run    
      --endPhase endPhase					  Last phase to run

