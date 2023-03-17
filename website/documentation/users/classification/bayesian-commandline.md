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
title: bayesian-commandline

    
---

# Naive Bayes commandline documentation

<a name="bayesian-commandline-Introduction"></a>
## Introduction

This quick start page describes how to run the naive bayesian and
complementary naive bayesian classification algorithms on a Hadoop cluster.

<a name="bayesian-commandline-Steps"></a>
## Steps

<a name="bayesian-commandline-Testingitononesinglemachinew/ocluster"></a>
### Testing it on one single machine w/o cluster

In the examples directory type:

    mvn -q exec:java
        -Dexec.mainClass="org.apache.mahout.classifier.bayes.mapreduce.bayes.<JOB>"
        -Dexec.args="<OPTIONS>"

    mvn -q exec:java
        -Dexec.mainClass="org.apache.mahout.classifier.bayes.mapreduce.cbayes.<JOB>"
        -Dexec.args="<OPTIONS>"


<a name="bayesian-commandline-Runningitonthecluster"></a>
### Running it on the cluster

* In $MAHOUT_HOME/, build the jar containing the job (mvn install) The job
will be generated in $MAHOUT_HOME/core/target/ and it's name will contain
the Mahout version number. For example, when using Mahout 0.1 release, the
job will be mahout-core-0.1.jar

* (Optional) 1 Start up Hadoop: $HADOOP_HOME/bin/start-all.sh

* Put the data: $HADOOP_HOME/bin/hadoop fs -put <PATH TO DATA> testdata

* Run the Job: $HADOOP_HOME/bin/hadoop jar

    $MAHOUT_HOME/core/target/mahout-core-<MAHOUT VERSION>.job
        org.apache.mahout.classifier.bayes.mapreduce.bayes.BayesDriver <OPTIONS>

* Get the data out of HDFS and have a look. Use bin/hadoop fs -lsr output
to view all outputs.

<a name="bayesian-commandline-Commandlineoptions"></a>
## Command line options

    BayesDriver, BayesThetaNormalizerDriver, CBayesNormalizedWeightDriver, CBayesDriver, CBayesThetaDriver, CBayesThetaNormalizerDriver, BayesWeightSummerDriver, BayesFeatureDriver, BayesTfIdfDriver Usage:
        [--input <input> --output <output> --help]
      
    Options
    
      --input (-i) input	  The Path for input Vectors. Must be a SequenceFile of Writable, Vector.
      --output (-o) output	  The directory pathname for output points.
      --help (-h)		  Print out help.

