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
title: Collocations

    
---


<a name="Collocations-CollocationsinMahout"></a>
# Collocations in Mahout

A collocation is defined as a sequence of words or terms which co-occur
more often than would be expected by chance. Statistically relevant
combinations of terms identify additional lexical units which can be
treated as features in a vector-based representation of a text. A detailed
discussion of collocations can be found on [Wikipedia](http://en.wikipedia.org/wiki/Collocation).

See there for a more detailed discussion of collocations in the [Reuters example](http://comments.gmane.org/gmane.comp.apache.mahout.user/5685).

<a name="Collocations-Log-LikelihoodbasedCollocationIdentification"></a>
## Theory behind implementation: Log-Likelihood based Collocation Identification

Mahout provides an implementation of a collocation identification algorithm
which scores collocations using log-likelihood ratio. The log-likelihood
score indicates the relative usefulness of a collocation with regards other
term combinations in the text. Collocations with the highest scores in a
particular corpus will generally be more useful as features.

Calculating the LLR is very straightforward and is described concisely in
[Ted Dunning's blog post](http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html)
. Ted describes the series of counts reqired to calculate the LLR for two
events A and B in order to determine if they co-occur more often than pure
chance. These counts include the number of times the events co-occur (k11),
the number of times the events occur without each other (k12 and k21), and
the number of times anything occurs. These counts are summarized in the
following table:

<table>
<tr><td> </td><td> Event A </td><td> Everything but Event A </td></tr>
<tr><td> Event B </td><td> A and B together (k11) </td><td>  B but not A (k12) </td></tr>
<tr><td> Everything but Event B </td><td> A but not B (k21) </td><td> Neither B nor A (k22) </td></tr>
</table>

For the purposes of collocation identification, it is useful to begin by
thinking in word pairs, bigrams. In this case the leading or head term from
the pair corresponds to A from the table above, B corresponds to the
trailing or tail term, while neither B nor A is the total number of word
pairs in the corpus less those containing B, A or both B and A.

Given the word pair of 'oscillation overthruster', the Log-Likelihood ratio
is computed by looking at the number of occurences of that word pair in the
corpus, the number of word pairs that begin with 'oscillation' but end with
something other than 'overthruster', the number of word pairs that end with
'overthruster' begin with something other than 'oscillation' and the number
of word pairs in the corpus that contain neither 'oscillation' and
overthruster.

This can be extended from bigrams to trigrams, 4-grams and beyond. In these
cases, the current algorithm uses the first token of the ngram as the head
of the ngram and the remaining n-1 tokens from the ngram, the n-1gram as it
were, as the tail. Given the trigram 'hong kong cavaliers', 'hong' is
treated as the head while 'kong cavaliers' is treated as the tail. Future
versions of this algorithm will allow for variations in which tokens of the
ngram are treated as the head and tail.

Beyond ngrams, it is often useful to inspect cases where individual words
occur around other interesting features of the text such as sentence
boundaries.

<a name="Collocations-GeneratingNGrams"></a>
## Generating NGrams

The tools that the collocation identification algorithm are embeeded within
either consume tokenized text as input or provide the ability to specify an
implementation of the Lucene Analyzer class perform tokenization in order
to form ngrams. The tokens are passed through a Lucene ShingleFilter to
produce NGrams of the desired length. 

Given the text "Alice was beginning to get very tired" as an example,
Lucene's StandardAnalyzer produces the tokens 'alice', 'beginning', 'get',
'very' and 'tired', while the ShingleFilter with a max NGram size set to 3
produces the shingles 'alice beginning', 'alice beginning get', 'beginning
get', 'beginning get very', 'get very', 'get very tired' and 'very tired'.
Note that both bigrams and trigrams are produced here. A future enhancement
to the existing algorithm would involve limiting the output to a particular
gram size as opposed to solely specifiying a max ngram size.

<a name="Collocations-RunningtheCollocationIdentificationAlgorithm."></a>
## Running the Collocation Identification Algorithm.

There are a couple ways to run the llr-based collocation algorithm in
mahout

<a name="Collocations-Whencreatingvectorsfromasequencefile"></a>
### When creating vectors from a sequence file

The llr collocation identifier is integrated into the process that is used
to create vectors from sequence files of text keys and values. Collocations
are generated when the --maxNGramSize (-ng) option is not specified and
defaults to 2 or is set to a number of 2 or greater. The --minLLR option
can be used to control the cutoff that prevents collocations below the
specified LLR score from being emitted, and the --minSupport argument can
be used to filter out collocations that appear below a certain number of
times. 


    bin/mahout seq2sparse
    
    Usage:									    
         [--minSupport <minSupport> --analyzerName <analyzerName> --chunkSize <chunkSize>
          --output <output> --input <input> --minDF <minDF>
          --maxDFPercent<maxDFPercent> --weight <weight> --norm <norm> --minLLR <minLLR>
          --numReducers  <numReducers> --maxNGramSize <ngramSize> --overwrite --help		    
          --sequentialAccessVector]
    Options 								    

      --minSupport (-s) minSupport	  (Optional) Minimum Support. Default Value: 2				    

      --analyzerName (-a) analyzerName    The class name of the analyzer

      --chunkSize (-chunk) chunkSize      The chunkSize in MegaBytes. 100-10000MB

      --output (-o) output		 The output directory

      --input (-i) input		   Input dir containing the documents in sequence file format

      --minDF (-md) minDF		  The minimum document frequency. Default is 1

      --maxDFPercent (-x) maxDFPercent    The max percentage of docs for the DF. Can be used to remove 
                                          really high frequency terms. Expressed as an
                                          integer between 0 and 100. Default is 99.     

      --weight (-wt) weight 	      The kind of weight to use. Currently TF   
    				      or TFIDF				    

      --norm (-n) norm		      The norm to use, expressed as either a    
    				      float or "INF" if you want to use the 
    				      Infinite norm.  Must be greater orequal  
    				      to 0.  The default is not to normalize    

      --minLLR (-ml) minLLR 	      (Optional)The minimum Log Likelihood  
    				      Ratio(Float)  Default is 1.0
	    
      --numReducers (-nr) numReducers     (Optional) Number of reduce tasks.    
    				      Default Value: 1			    

      --maxNGramSize (-ng) ngramSize      (Optional) The maximum size of ngrams to  
    				      create (2 = bigrams, 3 = trigrams, etc)   
    				      Default Value:2			 
   
      --overwrite (-w)		      If set, overwrite the output directory    
      --help (-h)			      Print out help			    
      --sequentialAccessVector (-seq)     (Optional) Whether output vectors should	
    				      be SequentialAccessVectors If set true	
    				      else false 


<a name="Collocations-CollocDriver"></a>
### CollocDriver


    bin/mahout org.apache.mahout.vectorizer.collocations.llr.CollocDriver
    
    Usage:									    
     [--input <input> --output <output> --maxNGramSize <ngramSize> --overwrite    
    --minSupport <minSupport> --minLLR <minLLR> --numReducers <numReducers>     
    --analyzerName <analyzerName> --preprocess --unigram --help]

    Options 								    

      --input (-i) input		      The Path for input files. 	    

      --output (-o) output		      The Path write output to		    

      --maxNGramSize (-ng) ngramSize      (Optional) The maximum size of ngramsto  
    				      create (2 = bigrams, 3 = trigrams,etc)   
    				      Default Value:2			
    
      --overwrite (-w)		      If set, overwrite the outputdirectory    

      --minSupport (-s) minSupport	      (Optional) Minimum Support. Default   
    				      Value: 2				    

      --minLLR (-ml) minLLR 	      (Optional)The minimum Log Likelihood
    				      Ratio(Float)  Default is 1.0	  
  
      --numReducers (-nr) numReducers     (Optional) Number of reduce tasks.    
    				      Default Value: 1			    

      --analyzerName (-a) analyzerName    The class name of the analyzer	    

      --preprocess (-p)		      If set, input is SequenceFile<Text,Text>  
    				      where the value is the document, which	
    				      will be tokenized using the specified 
    				      analyzer. 			
    
      --unigram (-u)		      If set, unigrams will be emitted inthe   
    				      final output alongside collocations
   
      --help (-h)			      Print out help	      


<a name="Collocations-Algorithmdetails"></a>
## Algorithm details

This section describes the implementation of the collocation identification
algorithm in terms of the map-reduce phases that are used to generate
ngrams and count the frequencies required to perform the log-likelihood
calculation. Unless otherwise noted, classes that are indicated in
CamelCase can be found in the mahout-utils module under the package
org.apache.mahout.utils.nlp.collocations.llr

The algorithm is implemented in two map-reduce passes:

<a name="Collocations-Pass1:CollocDriver.generateCollocations(...)"></a>
### Pass 1: CollocDriver.generateCollocations(...)

Generates NGrams and counts frequencies for ngrams, head and tail subgrams.

<a name="Collocations-Map:CollocMapper"></a>
#### Map: CollocMapper

Input k: Text (documentId), v: StringTuple (tokens) 

Each call to the mapper passes in the full set of tokens for the
corresponding document using a StringTuple. The ShingleFilter is run across
these tokens to produce ngrams of the desired length. ngrams and
frequencies are collected across the entire document.

Once this is done, ngrams are split into head and tail portions. A key of type GramKey is generated which is used later to join ngrams with their heads and tails in the reducer phase. The GramKey is a composite key made up of a string n-gram fragement as the primary key and a secondary key used for grouping and sorting in the reduce phase. The secondary key will either be EMPTY in the case where we are collecting either the head or tail of an ngram as the value or it will contain the byte[](.html)
 form of the ngram when collecting an ngram as the value.


    head_key(EMPTY) -> (head subgram, head frequency)

    head_key(ngram) -> (ngram, ngram frequency) 

    tail_key(EMPTY) -> (tail subgram, tail frequency)

    tail_key(ngram) -> (ngram, ngram frequency)


subgram and ngram values are packaged in Gram objects.

For each ngram found, the Count.NGRAM_TOTAL counter is incremented. When
the pass is complete, this counter will hold the total number of ngrams
encountered in the input which is used as a part of the LLR calculation.

Output k: GramKey (head or tail subgram), v: Gram (head, tail or ngram with
frequency)

<a name="Collocations-Combiner:CollocCombiner"></a>
#### Combiner: CollocCombiner

Input k: GramKey, v:Gram (as above)

This phase merges the counts for unique ngrams or ngram fragments across
multiple documents. The combiner treats the entire GramKey as the key and
as such, identical tuples from separate documents are passed into a single
call to the combiner's reduce method, their frequencies are summed and a
single tuple is passed out via the collector.

Output k: GramKey, v:Gram

<a name="Collocations-Reduce:CollocReducer"></a>
#### Reduce: CollocReducer

Input k: GramKey, v: Gram (as above)

The CollocReducer employs the Hadoop secondary sort strategy to avoid
caching ngram tuples in memory in order to calculate total ngram and
subgram frequencies. The GramKeyPartitioner ensures that tuples with the
same primary key are sent to the same reducer while the
GramKeyGroupComparator ensures that iterator provided by the reduce method
first returns the subgram and then returns ngram values grouped by ngram.
This eliminates the need to cache the values returned by the iterator in
order to calculate total frequencies for both subgrams and ngrams. There
input will consist of multiple frequencies for each (subgram_key, subgram)
or (subgram_key, ngram) tuple; one from each map task executed in which the
particular subgram was found.
The input will be traversed in the following order:


    (head subgram, frequency 1)
    (head subgram, frequency 2)
    ... 
    (head subgram, frequency N)
    (ngram 1, frequency 1)
    (ngram 1, frequency 2)
    ...
    (ngram 1, frequency N)
    (ngram 2, frequency 1)
    (ngram 2, frequency 2)
    ...
    (ngram 2, frequency N)
    ...
    (ngram N, frequency 1)
    (ngram N, frequency 2)
    ...
    (ngram N, frequency N)


Where all of the ngrams above share the same head. Data is presented in the
same manner for the tail subgrams.

As the values for a subgram or ngram are traversed, frequencies are
accumulated. Once all values for a subgram or ngram are processed the
resulting key/value pairs are passed to the collector as long as the ngram
frequency is equal to or greater than the specified minSupport. When an
ngram is skipped in this way the Skipped.LESS_THAN_MIN_SUPPORT counter to
be incremented.

Pairs are passed to the collector in the following format:


    ngram, ngram frequency -> subgram subgram frequency


In this manner, the output becomes an unsorted version of the following:


    ngram 1, frequency -> ngram 1 head, head frequency
    ngram 1, frequency -> ngram 1 tail, tail frequency
    ngram 2, frequency -> ngram 2 head, head frequency
    ngram 2, frequency -> ngram 2 tail, tail frequency
    ngram N, frequency -> ngram N head, head frequency
    ngram N, frequency -> ngram N tail, tail frequency


Output is in the format k:Gram (ngram, frequency), v:Gram (subgram,
frequency)

<a name="Collocations-Pass2:CollocDriver.computeNGramsPruneByLLR(...)"></a>
### Pass 2: CollocDriver.computeNGramsPruneByLLR(...)

Pass 1 has calculated full frequencies for ngrams and subgrams, Pass 2
performs the LLR calculation.

<a name="Collocations-MapPhase:IdentityMapper(org.apache.hadoop.mapred.lib.IdentityMapper)"></a>
#### Map Phase: IdentityMapper (org.apache.hadoop.mapred.lib.IdentityMapper)

This phase is a no-op. The data is passed through unchanged. The rest of
the work for llr calculation is done in the reduce phase.

<a name="Collocations-ReducePhase:LLRReducer"></a>
#### Reduce Phase: LLRReducer

Input is k:Gram, v:Gram (as above)

This phase receives the head and tail subgrams and their frequencies for
each ngram (with frequency) produced for the input:


    ngram 1, frequency -> ngram 1 head, frequency; ngram 1 tail, frequency
    ngram 2, frequency -> ngram 2 head, frequency; ngram 2 tail, frequency
    ...
    ngram 1, frequency -> ngram N head, frequency; ngram N tail, frequency


It also reads the full ngram count obtained from the first pass, passed in
as a configuration option. The parameters to the llr calculation are
calculated as follows:

k11 = f_n
k12 = f_h - f_n
k21 = f_t - f_n
k22 = N - ((f_h + f_t) - f_n)

Where f_n is the ngram frequency, f_h and f_t the frequency of head and
tail and N is the total number of ngrams.

Tokens with a llr below that of the specified minimum llr are dropped and
the Skipped.LESS_THAN_MIN_LLR counter is incremented.

Output is k: Text (ngram), v: DoubleWritable (llr score)

<a name="Collocations-Unigrampass-through."></a>
### Unigram pass-through.

By default in seq2sparse, or if the -u option is provided to the
CollocDriver, unigrams (single tokens) will be passed through the job and
each token's frequency will be calculated. As with ngrams, unigrams are
subject to filtering with minSupport and minLLR.

