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
title: Cluster Dumper

   
---

<a name="ClusterDumper-Introduction"></a>
## Cluster Dumper - Introduction

Clustering tasks in Mahout will output data in the format of a SequenceFile
(Text, Cluster) and the Text is a cluster identifier string. To analyze
this output we need to convert the sequence files to a human readable
format and this is achieved using the clusterdump utility.

<a name="ClusterDumper-Stepsforanalyzingclusteroutputusingclusterdumputility"></a>
## Steps for analyzing cluster output using clusterdump utility

After you've executed a clustering tasks (either examples or real-world),
you can run clusterdumper in 2 modes:


1. Hadoop Environment
1. Standalone Java Program 


<a name="ClusterDumper-HadoopEnvironment{anchor:HadoopEnvironment}"></a>
### Hadoop Environment

If you have setup your HADOOP_HOME environment variable, you can use the
command line utility `mahout` to execute the ClusterDumper on Hadoop. In
this case we wont need to get the output clusters to our local machines.
The utility will read the output clusters present in HDFS and output the
human-readable cluster values into our local file system. Say you've just
executed the [synthetic control example ](clustering-of-synthetic-control-data.html)
 and want to analyze the output, you can execute the `mahout clusterdumper` utility from the command line.

#### CLI options:
    --help                               Print out help	
    --input (-i) input                   The directory containing Sequence
                                           Files for the Clusters	    
    --output (-o) output                 The output file.  If not specified,
                                           dumps to the console.
    --outputFormat (-of) outputFormat    The optional output format to write
                                           the results as. Options: TEXT, CSV, or GRAPH_ML		 
    --substring (-b) substring           The number of chars of the	    
    					   asFormatString() to print	
    --pointsDir (-p) pointsDir           The directory containing points  
                                           sequence files mapping input vectors
                                           to their cluster.  If specified, 
                                           then the program will output the 
                                           points associated with a cluster 
    --dictionary (-d) dictionary         The dictionary file.
    --dictionaryType (-dt) dictionaryType    The dictionary file type	    
                                         (text|sequencefile)
    --distanceMeasure (-dm) distanceMeasure  The classname of the DistanceMeasure.
                                               Default is SquaredEuclidean.
    --numWords (-n) numWords             The number of top terms to print 
    --tempDir tempDir                    Intermediate output directory
    --startPhase startPhase              First phase to run
    --endPhase endPhase                  Last phase to run
    --evaluate (-e)                      Run ClusterEvaluator and CDbwEvaluator over the
                                          input. The output will be appended to the rest of
                                          the output at the end.   

### Standalone Java Program                                          

Run the clusterdump utility as follows as a standalone Java Program through Eclipse. <!-- - if you are using eclipse, setup mahout-utils as a project as specified in [Working with Maven in Eclipse](../../developers/buildingmahout.html). -->
    To execute ClusterDumper.java,
    
* Under mahout-utils, Right-Click on ClusterDumper.java
* Choose Run-As, Run Configurations
* On the left menu, click on Java Application
* On the top-bar click on "New Launch Configuration"
* A new launch should be automatically created with project as

    "mahout-utils" and Main Class as "org.apache.mahout.utils.clustering.ClusterDumper"

In the arguments tab, specify the below arguments


    --seqFileDir <MAHOUT_HOME>/examples/output/clusters-10 
    --pointsDir <MAHOUT_HOME>/examples/output/clusteredPoints 
    --output <MAHOUT_HOME>/examples/output/clusteranalyze.txt
    replace <MAHOUT_HOME> with the actual path of your $MAHOUT_HOME

* Hit run to execute the ClusterDumper using Eclipse. Setting breakpoints etc should just work fine.
    
Reading the output file
    
This will output the clusters into a file called clusteranalyze.txt inside $MAHOUT_HOME/examples/output
Sample data will look like

CL-0 { n=116 c=[29.922, 30.407, 30.373, 30.094, 29.886, 29.937, 29.751, 30.054, 30.039, 30.126, 29.764, 29.835, 30.503, 29.876, 29.990, 29.605, 29.379, 30.120, 29.882, 30.161, 29.825, 30.074, 30.001, 30.421, 29.867, 29.736, 29.760, 30.192, 30.134, 30.082, 29.962, 29.512, 29.736, 29.594, 29.493, 29.761, 29.183, 29.517, 29.273, 29.161, 29.215, 29.731, 29.154, 29.113, 29.348, 28.981, 29.543, 29.192, 29.479, 29.406, 29.715, 29.344, 29.628, 29.074, 29.347, 29.812, 29.058, 29.177, 29.063, 29.607](29.922,-30.407,-30.373,-30.094,-29.886,-29.937,-29.751,-30.054,-30.039,-30.126,-29.764,-29.835,-30.503,-29.876,-29.990,-29.605,-29.379,-30.120,-29.882,-30.161,-29.825,-30.074,-30.001,-30.421,-29.867,-29.736,-29.760,-30.192,-30.134,-30.082,-29.962,-29.512,-29.736,-29.594,-29.493,-29.761,-29.183,-29.517,-29.273,-29.161,-29.215,-29.731,-29.154,-29.113,-29.348,-28.981,-29.543,-29.192,-29.479,-29.406,-29.715,-29.344,-29.628,-29.074,-29.347,-29.812,-29.058,-29.177,-29.063,-29.607.html)
 r=[3.463, 3.351, 3.452, 3.438, 3.371, 3.569, 3.253, 3.531, 3.439, 3.472,
3.402, 3.459, 3.320, 3.260, 3.430, 3.452, 3.320, 3.499, 3.302, 3.511,
3.520, 3.447, 3.516, 3.485, 3.345, 3.178, 3.492, 3.434, 3.619, 3.483,
3.651, 3.833, 3.812, 3.433, 4.133, 3.855, 4.123, 3.999, 4.467, 4.731,
4.539, 4.956, 4.644, 4.382, 4.277, 4.918, 4.784, 4.582, 4.915, 4.607,
4.672, 4.577, 5.035, 5.241, 4.731, 4.688, 4.685, 4.657, 4.912, 4.300] }

and on...

where CL-0 is the Cluster 0 and n=116 refers to the number of points observed by this cluster and c = \[29.922 ...\]
 refers to the center of Cluster as a vector and r = \[3.463 ..\] refers to
the radius of the cluster as a vector.