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
title: (Deprecated)  Wikipedia XML parser and Naive Bayes Example

    
---
# Wikipedia XML parser and Naive Bayes Classifier Example

## Introduction
Mahout has an [example script](https://github.com/apache/mahout/blob/master/examples/bin/classify-wikipedia.sh) [1] which will download a recent XML dump of the (entire if desired) [English Wikipedia database](http://dumps.wikimedia.org/enwiki/latest/). After running the classification script, you can use the [document classification script](https://github.com/apache/mahout/blob/master/examples/bin/spark-document-classifier.mscala) from the Mahout [spark-shell](http://mahout.apache.org/users/sparkbindings/play-with-shell.html) to vectorize and classify text from outside of the training and testing corpus using a modle built on the Wikipedia dataset.  

You can run this script to build and test a Naive Bayes classifier for option (1) 10 arbitrary countries or option (2) 2 countries (United States and United Kingdom).

## Oververview

Tou run the example simply execute the `$MAHOUT_HOME/examples/bin/classify-wikipedia.sh` script.

By defult the script is set to run on a medium sized Wikipedia XML dump.  To run on the full set (the entire english Wikipedia) you can change the download by commenting out line 78, and uncommenting line 80  of [classify-wikipedia.sh](https://github.com/apache/mahout/blob/master/examples/bin/classify-wikipedia.sh) [1]. However this is not recommended unless you have the resources to do so. *Be sure to clean your work directory when changing datasets- option (3).*

The step by step process for Creating a Naive Bayes Classifier for the Wikipedia XML dump is very similar to that for [creating a 20 Newsgroups Classifier](http://mahout.apache.org/users/classification/twenty-newsgroups.html) [4].  The only difference being that instead of running `$mahout seqdirectory` on the unzipped 20 Newsgroups file, you'll run `$mahout seqwiki` on the unzipped Wikipedia xml dump.

    $ mahout seqwiki 

The above command launches `WikipediaToSequenceFile.java` which accepts a text file of categories [3] and starts an MR job to parse the each document in the XML file.  This process will seek to extract documents with a wikipedia category tag which (exactly, if the `-exactMatchOnly` option is set) matches a line in the category file.  If no match is found and the `-all` option is set, the document will be dumped into an "unknown" category. The documents will then be written out as a `<Text,Text>` sequence file of the form (K:/category/document_title , V: document).

There are 3 different example category files available to in the /examples/src/test/resources
directory:  country.txt, country10.txt and country2.txt.  You can edit these categories to extract a different corpus from the Wikipedia dataset.

The CLI options for `seqwiki` are as follows:

    --input          (-i)         input pathname String
    --output         (-o)         the output pathname String
    --categories     (-c)         the file containing the Wikipedia categories
    --exactMatchOnly (-e)         if set, then the Wikipedia category must match
                                    exactly instead of simply containing the category string
    --all            (-all)       if set select all categories
    --removeLabels   (-rl)        if set, remove [[Category:labels]] from document text after extracting label.


After `seqwiki`, the script runs `seq2sparse`, `split`, `trainnb` and `testnb` as in the [step by step 20newsgroups example](http://mahout.apache.org/users/classification/twenty-newsgroups.html).  When all of the jobs have finished, a confusion matrix will be displayed.

#Resourcese

[1] [classify-wikipedia.sh](https://github.com/apache/mahout/blob/master/examples/bin/classify-wikipedia.sh)

[2] [Document classification script for the Mahout Spark Shell](https://github.com/apache/mahout/blob/master/examples/bin/spark-document-classifier.mscala)

[3] [Example category file](https://github.com/apache/mahout/blob/master/examples/src/test/resources/country10.txt)

[4] [Step by step instructions for building a Naive Bayes classifier for 20newsgroups from the command line](http://mahout.apache.org/users/classification/twenty-newsgroups.html)

[5] [Mahout MapReduce Naive Bayes](http://mahout.apache.org/users/classification/bayesian.html)

[6] [Mahout Spark Naive Bayes](http://mahout.apache.org/users/algorithms/spark-naive-bayes.html)

[7] [Mahout Scala Spark and H2O Bindings](http://mahout.apache.org/users/sparkbindings/home.html)

