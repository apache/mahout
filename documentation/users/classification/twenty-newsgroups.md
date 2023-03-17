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
title: Twenty Newsgroups

    
---


<a name="TwentyNewsgroups-TwentyNewsgroupsClassificationExample"></a>
## Twenty Newsgroups Classification Example

<a name="TwentyNewsgroups-Introduction"></a>
## Introduction

The 20 newsgroups dataset is a collection of approximately 20,000
newsgroup documents, partitioned (nearly) evenly across 20 different
newsgroups. The 20 newsgroups collection has become a popular data set for
experiments in text applications of machine learning techniques, such as
text classification and text clustering. We will use the [Mahout CBayes](http://mahout.apache.org/users/mapreduce/classification/bayesian.html)
classifier to create a model that would classify a new document into one of
the 20 newsgroups.

<a name="TwentyNewsgroups-Prerequisites"></a>
### Prerequisites

* Mahout has been downloaded ([instructions here](https://mahout.apache.org/general/downloads.html))
* Maven is available
* Your environment has the following variables:
     * **HADOOP_HOME** Environment variables refers to where Hadoop lives 
     * **MAHOUT_HOME** Environment variables refers to where Mahout lives

<a name="TwentyNewsgroups-Instructionsforrunningtheexample"></a>
### Instructions for running the example

1. If running Hadoop in cluster mode, start the hadoop daemons by executing the following commands:

            $ cd $HADOOP_HOME/bin
            $ ./start-all.sh
   
    Otherwise:

            $ export MAHOUT_LOCAL=true

2. In the trunk directory of Mahout, compile and install Mahout:

            $ cd $MAHOUT_HOME
            $ mvn -DskipTests clean install

3. Run the [20 newsgroups example script](https://github.com/apache/mahout/blob/master/examples/bin/classify-20newsgroups.sh) by executing:

            $ ./examples/bin/classify-20newsgroups.sh

4. You will be prompted to select a classification method algorithm: 
    
            1. Complement Naive Bayes
            2. Naive Bayes
            3. Stochastic Gradient Descent

Select 1 and the the script will perform the following:

1. Create a working directory for the dataset and all input/output.
2. Download and extract the *20news-bydate.tar.gz* from the [20 newsgroups dataset](http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz) to the working directory.
3. Convert the full 20 newsgroups dataset into a < Text, Text > SequenceFile. 
4. Convert and preprocesses the dataset into a < Text, VectorWritable > SequenceFile containing term frequencies for each document.
5. Split the preprocessed dataset into training and testing sets. 
6. Train the classifier.
7. Test the classifier.


Output should look something like:


    =======================================================
    Confusion Matrix
    -------------------------------------------------------
     a  b  c  d  e  f  g  h  i  j  k  l  m  n  o  p  q  r  s  t <--Classified as
    381 0  0  0  0  9  1  0  0  0  1  0  0  2  0  1  0  0  3  0 |398 a=rec.motorcycles
     1 284 0  0  0  0  1  0  6  3  11 0  66 3  0  6  0  4  9  0 |395 b=comp.windows.x
     2  0 339 2  0  3  5  1  0  0  0  0  1  1  12 1  7  0  2  0 |376 c=talk.politics.mideast
     4  0  1 327 0  2  2  0  0  2  1  1  0  5  1  4  12 0  2  0 |364 d=talk.politics.guns
     7  0  4  32 27 7  7  2  0  12 0  0  6  0 100 9  7  31 0  0 |251 e=talk.religion.misc
     10 0  0  0  0 359 2  2  0  0  3  0  1  6  0  1  0  0  11 0 |396 f=rec.autos
     0  0  0  0  0  1 383 9  1  0  0  0  0  0  0  0  0  3  0  0 |397 g=rec.sport.baseball
     1  0  0  0  0  0  9 382 0  0  0  0  1  1  1  0  2  0  2  0 |399 h=rec.sport.hockey
     2  0  0  0  0  4  3  0 330 4  4  0  5  12 0  0  2  0  12 7 |385 i=comp.sys.mac.hardware
     0  3  0  0  0  0  1  0  0 368 0  0  10 4  1  3  2  0  2  0 |394 j=sci.space
     0  0  0  0  0  3  1  0  27 2 291 0  11 25 0  0  1  0  13 18|392 k=comp.sys.ibm.pc.hardware
     8  0  1 109 0  6  11 4  1  18 0  98 1  3  11 10 27 1  1  0 |310 l=talk.politics.misc
     0  11 0  0  0  3  6  0  10 6  11 0 299 13 0  2  13 0  7  8 |389 m=comp.graphics
     6  0  1  0  0  4  2  0  5  2  12 0  8 321 0  4  14 0  8  6 |393 n=sci.electronics
     2  0  0  0  0  0  4  1  0  3  1  0  3  1 372 6  0  2  1  2 |398 o=soc.religion.christian
     4  0  0  1  0  2  3  3  0  4  2  0  7  12 6 342 1  0  9  0 |396 p=sci.med
     0  1  0  1  0  1  4  0  3  0  1  0  8  4  0  2 369 0  1  1 |396 q=sci.crypt
     10 0  4  10 1  5  6  2  2  6  2  0  2  1 86 15 14 152 0  1 |319 r=alt.atheism
     4  0  0  0  0  9  1  1  8  1  12 0  3  0  2  0  0  0 341 2 |390 s=misc.forsale
     8  5  0  0  0  1  6  0  8  5  50 0  40 2  1  0  9  0  3 256|394 t=comp.os.ms-windows.misc
    =======================================================
    Statistics
    -------------------------------------------------------
    Kappa                                       0.8808
    Accuracy                                   90.8596%
    Reliability                                86.3632%
    Reliability (standard deviation)            0.2131





<a name="TwentyNewsgroups-ComplementaryNaiveBayes"></a>
## End to end commands to build a CBayes model for 20 newsgroups
The [20 newsgroups example script](https://github.com/apache/mahout/blob/master/examples/bin/classify-20newsgroups.sh) issues the following commands as outlined above. We can build a CBayes classifier from the command line by following the process in the script: 

*Be sure that **MAHOUT_HOME**/bin and **HADOOP_HOME**/bin are in your **$PATH***

1. Create a working directory for the dataset and all input/output.
           
            $ export WORK_DIR=/tmp/mahout-work-${USER}
            $ mkdir -p ${WORK_DIR}

2. Download and extract the *20news-bydate.tar.gz* from the [20newsgroups dataset](http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz) to the working directory.

            $ curl http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz 
                -o ${WORK_DIR}/20news-bydate.tar.gz
            $ mkdir -p ${WORK_DIR}/20news-bydate
            $ cd ${WORK_DIR}/20news-bydate && tar xzf ../20news-bydate.tar.gz && cd .. && cd ..
            $ mkdir ${WORK_DIR}/20news-all
            $ cp -R ${WORK_DIR}/20news-bydate/*/* ${WORK_DIR}/20news-all
     * If you're running on a Hadoop cluster:
 
            $ hadoop dfs -put ${WORK_DIR}/20news-all ${WORK_DIR}/20news-all

3. Convert the full 20 newsgroups dataset into a < Text, Text > SequenceFile. 
          
            $ mahout seqdirectory 
                -i ${WORK_DIR}/20news-all 
                -o ${WORK_DIR}/20news-seq 
                -ow
            
4. Convert and preprocesses the dataset into  a < Text, VectorWritable > SequenceFile containing term frequencies for each document. 
            
            $ mahout seq2sparse 
                -i ${WORK_DIR}/20news-seq 
                -o ${WORK_DIR}/20news-vectors
                -lnorm 
                -nv 
                -wt tfidf
If we wanted to use different parsing methods or transformations on the term frequency vectors we could supply different options here e.g.: -ng 2 for bigrams or -n 2 for L2 length normalization.  See the [Creating vectors from text](http://mahout.apache.org/users/basics/creating-vectors-from-text.html) page for a list of all seq2sparse options.   

5. Split the preprocessed dataset into training and testing sets.

            $ mahout split 
                -i ${WORK_DIR}/20news-vectors/tfidf-vectors 
                --trainingOutput ${WORK_DIR}/20news-train-vectors 
                --testOutput ${WORK_DIR}/20news-test-vectors  
                --randomSelectionPct 40 
                --overwrite --sequenceFiles -xm sequential
 
6. Train the classifier.

            $ mahout trainnb 
                -i ${WORK_DIR}/20news-train-vectors
                -el  
                -o ${WORK_DIR}/model 
                -li ${WORK_DIR}/labelindex 
                -ow 
                -c

7. Test the classifier.

            $ mahout testnb 
                -i ${WORK_DIR}/20news-test-vectors
                -m ${WORK_DIR}/model 
                -l ${WORK_DIR}/labelindex 
                -ow 
                -o ${WORK_DIR}/20news-testing 
                -c

 
       