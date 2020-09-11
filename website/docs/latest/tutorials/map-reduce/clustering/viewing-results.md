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
title: (Deprecated)  Viewing Results

   
---
<a name="ViewingResults-Intro"></a>
# Intro

Many of the Mahout libraries run as batch jobs, dumping results into Hadoop
sequence files or other data structures.  This page is intended to
demonstrate the various ways one might inspect the outcome of various jobs.
 The page is organized by algorithms.

<a name="ViewingResults-GeneralUtilities"></a>
# General Utilities

<a name="ViewingResults-SequenceFileDumper"></a>
## Sequence File Dumper


<a name="ViewingResults-Clustering"></a>
# Clustering

<a name="ViewingResults-ClusterDumper"></a>
## Cluster Dumper

Run the following to print out all options:

    java  -cp "*" org.apache.mahout.utils.clustering.ClusterDumper --help



<a name="ViewingResults-Example"></a>
### Example

    java  -cp "*" org.apache.mahout.utils.clustering.ClusterDumper --seqFileDir
./solr-clust-n2/out/clusters-2
          --dictionary ./solr-clust-n2/dictionary.txt
          --substring 100 --pointsDir ./solr-clust-n2/out/points/
    



<a name="ViewingResults-ClusterLabels(MAHOUT-163)"></a>
## Cluster Labels (MAHOUT-163)

<a name="ViewingResults-Classification"></a>
# Classification
