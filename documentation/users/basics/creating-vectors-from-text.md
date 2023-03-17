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
title: Creating Vectors from Text

    
---


# Creating vectors from text
<a name="CreatingVectorsfromText-Introduction"></a>
# Introduction

For clustering and classifying documents it is usually necessary to convert the raw text
into vectors that can then be consumed by the clustering [Algorithms](algorithms.html).  These approaches are described below.

<a name="CreatingVectorsfromText-FromLucene"></a>
# From Lucene

*NOTE: Your Lucene index must be created with the same version of Lucene
used in Mahout.  As of Mahout 0.9 this is Lucene 4.6.1. If these versions dont match you will likely get "Exception in thread "main"
org.apache.lucene.index.CorruptIndexException: Unknown format version: -11"
as an error.*

Mahout has utilities that allow one to easily produce Mahout Vector
representations from a Lucene (and Solr, since they are they same) index.

For this, we assume you know how to build a Lucene/Solr index.	For those
who don't, it is probably easiest to get up and running using [Solr](http://lucene.apache.org/solr)
 as it can ingest things like PDFs, XML, Office, etc. and create a Lucene
index.	For those wanting to use just Lucene, see the [Lucene website](http://lucene.apache.org/core)
 or check out _Lucene In Action_ by Erik Hatcher, Otis Gospodnetic and Mike
McCandless.

To get started, make sure you get a fresh copy of Mahout from [GitHub](http://mahout.apache.org/developers/buildingmahout.html)
 and are comfortable building it. It defines interfaces and implementations
for efficiently iterating over a data source (it only supports Lucene
currently, but should be extensible to databases, Solr, etc.) and produces
a Mahout Vector file and term dictionary which can then be used for
clustering.   The main code for driving this is the driver program located
in the org.apache.mahout.utils.vectors package.  The driver program offers
several input options, which can be displayed by specifying the --help
option.  Examples of running the driver are included below:

<a name="CreatingVectorsfromText-GeneratinganoutputfilefromaLuceneIndex"></a>
#### Generating an output file from a Lucene Index


    $MAHOUT_HOME/bin/mahout lucene.vector 
        --dir (-d) dir                     The Lucene directory      
        --idField idField                  The field in the index    
                                               containing the index.  If 
                                               null, then the Lucene     
                                               internal doc id is used   
                                               which is prone to error   
                                               if the underlying index   
                                               changes                   
        --output (-o) output               The output file           
        --delimiter (-l) delimiter         The delimiter for         
                                               outputting the dictionary 
        --help (-h)                        Print out help            
        --field (-f) field                 The field in the index    
        --max (-m) max                         The maximum number of     
                                               vectors to output.  If    
                                               not specified, then it    
                                               will loop over all docs   
        --dictOut (-t) dictOut             The output of the         
                                               dictionary                
        --seqDictOut (-st) seqDictOut      The output of the         
                                               dictionary as sequence    
                                               file                      
        --norm (-n) norm                   The norm to use,          
                                               expressed as either a     
                                               double or "INF" if you    
                                               want to use the Infinite  
                                               norm.  Must be greater or 
                                               equal to 0.  The default  
                                               is not to normalize       
        --maxDFPercent (-x) maxDFPercent   The max percentage of     
                                               docs for the DF.  Can be  
                                               used to remove really     
                                               high frequency terms.     
                                               Expressed as an integer   
                                               between 0 and 100.        
                                               Default is 99.            
        --weight (-w) weight               The kind of weight to     
                                               use. Currently TF or      
                                               TFIDF                     
        --minDF (-md) minDF                The minimum document      
                                               frequency.  Default is 1  
        --maxPercentErrorDocs (-err) mErr  The max percentage of     
                                               docs that can have a null 
                                               term vector. These are    
                                               noise document and can    
                                               occur if the analyzer     
                                               used strips out all terms 
                                               in the target field. This 
                                               percentage is expressed   
                                               as a value between 0 and  
                                               1. The default is 0.  
  
#### Example: Create 50 Vectors from an Index 

    $MAHOUT_HOME/bin/mahout lucene.vector
        --dir $WORK_DIR/wikipedia/solr/data/index 
        --field body 
        --dictOut $WORK_DIR/solr/wikipedia/dict.txt
        --output $WORK_DIR/solr/wikipedia/out.txt 
        --max 50


This uses the index specified by --dir and the body field in it and writes
out the info to the output dir and the dictionary to dict.txt.	It only
outputs 50 vectors.  If you don't specify --max, then all the documents in
the index are output.

<a name="CreatingVectorsfromText-50VectorsFromLuceneL2Norm"></a>
#### Example: Creating 50 Normalized Vectors from a Lucene Index using the [L_2 Norm](http://en.wikipedia.org/wiki/Lp_space)

    $MAHOUT_HOME/bin/mahout lucene.vector 
        --dir $WORK_DIR/wikipedia/solr/data/index 
        --field body 
        --dictOut $WORK_DIR/solr/wikipedia/dict.txt
        --output $WORK_DIR/solr/wikipedia/out.txt 
        --max 50 
        --norm 2


<a name="CreatingVectorsfromText-FromDirectoryofTextdocuments"></a>
## From A Directory of Text documents
Mahout has utilities to generate Vectors from a directory of text
documents. Before creating the vectors, you need to convert the documents
to SequenceFile format. SequenceFile is a hadoop class which allows us to
write arbitary (key, value) pairs into it. The DocumentVectorizer requires the
key to be a Text with a unique document id, and value to be the Text
content in UTF-8 format.

You may find [Tika](http://tika.apache.org/) helpful in converting
binary documents to text.

<a name="CreatingVectorsfromText-ConvertingdirectoryofdocumentstoSequenceFileformat"></a>
#### Converting directory of documents to SequenceFile format
Mahout has a nifty utility which reads a directory path including its
sub-directories and creates the SequenceFile in a chunked manner for us.

    $MAHOUT_HOME/bin/mahout seqdirectory 
        --input (-i) input                       Path to job input directory.   
        --output (-o) output                     The directory pathname for     
                                                     output.                        
        --overwrite (-ow)                        If present, overwrite the      
                                                     output directory before        
                                                     running job                    
        --method (-xm) method                    The execution method to use:   
                                                     sequential or mapreduce.       
                                                     Default is mapreduce           
        --chunkSize (-chunk) chunkSize           The chunkSize in MegaBytes.    
                                                     Defaults to 64                 
        --fileFilterClass (-filter) fFilterClass The name of the class to use   
                                                     for file parsing. Default:     
                                                     org.apache.mahout.text.PrefixAdditionFilter                   
        --keyPrefix (-prefix) keyPrefix          The prefix to be prepended to  
                                                     the key                        
        --charset (-c) charset                   The name of the character      
                                                     encoding of the input files.   
                                                     Default to UTF-8 {accepts: cp1252|ascii...}             
        --method (-xm) method                    The execution method to use:   
                                                     sequential or mapreduce.       
                                                 Default is mapreduce           
        --overwrite (-ow)                        If present, overwrite the      
                                                     output directory before        
                                                     running job                    
        --help (-h)                              Print out help                 
        --tempDir tempDir                        Intermediate output directory  
        --startPhase startPhase                  First phase to run             
        --endPhase endPhase                      Last phase to run  

The output of seqDirectory will be a Sequence file < Text, Text > of all documents (/sub-directory-path/documentFileName, documentText).

<a name="CreatingVectorsfromText-CreatingVectorsfromSequenceFile"></a>
#### Creating Vectors from SequenceFile

From the sequence file generated from the above step run the following to
generate vectors. 


    $MAHOUT_HOME/bin/mahout seq2sparse
        --minSupport (-s) minSupport      (Optional) Minimum Support. Default       
                                              Value: 2                                  
        --analyzerName (-a) analyzerName  The class name of the analyzer            
        --chunkSize (-chunk) chunkSize    The chunkSize in MegaBytes. Default       
                                              Value: 100MB                              
        --output (-o) output              The directory pathname for output.        
        --input (-i) input                Path to job input directory.              
        --minDF (-md) minDF               The minimum document frequency.  Default  
                                              is 1                                      
        --maxDFSigma (-xs) maxDFSigma     What portion of the tf (tf-idf) vectors   
                                              to be used, expressed in times the        
                                              standard deviation (sigma) of the         
                                              document frequencies of these vectors.    
                                              Can be used to remove really high         
                                              frequency terms. Expressed as a double    
                                              value. Good value to be specified is 3.0. 
                                              In case the value is less than 0 no       
                                              vectors will be filtered out. Default is  
                                              -1.0.  Overrides maxDFPercent             
        --maxDFPercent (-x) maxDFPercent  The max percentage of docs for the DF.    
                                              Can be used to remove really high         
                                              frequency terms. Expressed as an integer  
                                              between 0 and 100. Default is 99.  If     
                                              maxDFSigma is also set, it will override  
                                              this value.                               
        --weight (-wt) weight             The kind of weight to use. Currently TF   
                                              or TFIDF. Default: TFIDF                  
        --norm (-n) norm                  The norm to use, expressed as either a    
                                              float or "INF" if you want to use the     
                                              Infinite norm.  Must be greater or equal  
                                              to 0.  The default is not to normalize    
        --minLLR (-ml) minLLR             (Optional)The minimum Log Likelihood      
                                              Ratio(Float)  Default is 1.0              
        --numReducers (-nr) numReducers   (Optional) Number of reduce tasks.        
                                              Default Value: 1                          
        --maxNGramSize (-ng) ngramSize    (Optional) The maximum size of ngrams to  
                                              create (2 = bigrams, 3 = trigrams, etc)   
                                              Default Value:1                           
        --overwrite (-ow)                 If set, overwrite the output directory    
        --help (-h)                           Print out help                            
        --sequentialAccessVector (-seq)   (Optional) Whether output vectors should  
                                              be SequentialAccessVectors. Default is false;
                                              true required for running some algorithms
                                              (LDA,Lanczos)                                
        --namedVector (-nv)               (Optional) Whether output vectors should  
                                              be NamedVectors. If set true else false   
        --logNormalize (-lnorm)           (Optional) Whether output vectors should  
                                              be logNormalize. If set true else false



This will create SequenceFiles of tokenized documents < Text, StringTuple >  (docID, tokenizedDoc) and vectorized documents < Text, VectorWritable > (docID, TF-IDF Vector).  

As well, seq2sparse will create SequenceFiles for: a dictionary (wordIndex, word), a word frequency count (wordIndex, count) and a document frequency count (wordIndex, DFCount) in the output directory. 

The --minSupport option is the min frequency for the word to be considered as a feature; --minDF is the min number of documents the word needs to be in; --maxDFPercent is the max value of the expression (document frequency of a word/total number of document) to be considered as good feature to be in the document. These options are helpful in removing high frequency features like stop words.

The vectorized documents can then be used as input to many of Mahout's classification and clustering algorithms.

#### Example: Creating Normalized [TF-IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) Vectors from a directory of text documents using [trigrams](http://en.wikipedia.org/wiki/N-gram) and the [L_2 Norm](http://en.wikipedia.org/wiki/Lp_space)
Create sequence files from the directory of text documents:
    
    $MAHOUT_HOME/bin/mahout seqdirectory 
        -i $WORK_DIR/reuters 
        -o $WORK_DIR/reuters-seqdir 
        -c UTF-8
        -chunk 64
        -xm sequential

Vectorize the documents using trigrams, L_2 length normalization and a maximum document frequency cutoff of 85%.

    $MAHOUT_HOME/bin/mahout seq2sparse 
        -i $WORK_DIR/reuters-out-seqdir/ 
        -o $WORK_DIR/reuters-out-seqdir-sparse-kmeans 
        --namedVec
        -wt tfidf
        -ng 3
        -n 2
        --maxDFPercent 85 

The sequence file in the $WORK_DIR/reuters-out-seqdir-sparse-kmeans/tfidf-vectors directory can now be used as input to the Mahout [k-Means](http://mahout.apache.org/users/clustering/k-means-clustering.html) clustering algorithm.

<a name="CreatingVectorsfromText-Background"></a>
## Background

* [Discussion on centroid calculations with sparse vectors](http://markmail.org/thread/l5zi3yk446goll3o)

<a name="CreatingVectorsfromText-ConvertingexistingvectorstoMahout'sformat"></a>
## Converting existing vectors to Mahout's format

If you are in the happy position to already own a document (as in: texts,
images or whatever item you wish to treat) processing pipeline, the
question arises of how to convert the vectors into the Mahout vector
format. Probably the easiest way to go would be to implement your own
Iterable<Vector> (called VectorIterable in the example below) and then
reuse the existing VectorWriter classes:


    VectorWriter vectorWriter = SequenceFile.createWriter(filesystem,
                                                          configuration,
                                                          outfile,
                                                          LongWritable.class,
                                                          SparseVector.class);

    long numDocs = vectorWriter.write(new VectorIterable(), Long.MAX_VALUE);

