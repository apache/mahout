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
title: (Deprecated)  Parallel Frequent Pattern Mining

    
---
Mahout has a Top K Parallel FPGrowth Implementation. Its based on the paper [http://infolab.stanford.edu/~echang/recsys08-69.pdf](http://infolab.stanford.edu/~echang/recsys08-69.pdf)
 with some optimisations in mining the data.

Given a huge transaction list, the algorithm finds all unique features(sets
of field values) and eliminates those features whose frequency in the whole
dataset is less that minSupport. Using these remaining features N, we find
the top K closed patterns for each of them, generating a total of NxK
patterns. FPGrowth Algorithm is a generic implementation, we can use any
Object type to denote a feature. Current implementation requires you to use
a String as the object type. You may implement a version for any object by
creating Iterators, Convertors and TopKPatternWritable for that particular
object. For more information please refer the package
org.apache.mahout.fpm.pfpgrowth.convertors.string

    e.g:
     FPGrowth<String> fp = new FPGrowth<String>();
     Set<String> features = new HashSet<String>();
     fp.generateTopKStringFrequentPatterns(
         new StringRecordIterator(new FileLineIterable(new File(input),
encoding, false), pattern),
    	fp.generateFList(
    	  new StringRecordIterator(new FileLineIterable(new File(input),
encoding, false), pattern), minSupport),
    	 minSupport,
    	maxHeapSize,
    	features,
    	new StringOutputConvertor(new SequenceFileOutputCollector<Text,
TopKStringPatterns>(writer))
      );

* The first argument is the iterator of transaction in this case its
Iterator<List<String>>
* The second argument is the output of generateFList function, which
returns the frequent items and their frequencies from the given database
transaction iterator
* The third argument is the minimum Support of the pattern to be generated
* The fourth argument is the maximum number of patterns to be mined for
each feature
* The fifth argument is the set of features for which the frequent patterns
has to be mined
* The last argument is an output collector which takes \[key, value\](key,-value\.html)
 of Feature and TopK Patterns of the format \[String,
List<Pair<List<String>, Long>>\] and writes them to the appropriate writer
class which takes care of storing the object, in this case in a Sequence
File Output format

<a name="ParallelFrequentPatternMining-RunningFrequentPatternGrowthviacommandline"></a>
## Running Frequent Pattern Growth via command line

The command line launcher for string transaction data
org.apache.mahout.fpm.pfpgrowth.FPGrowthDriver has other features including
specifying the regex pattern for spitting a string line of a transaction
into the constituent features.

Input files have to be in the following format.

<optional document id>TAB<TOKEN1>SPACE<TOKEN2>SPACE....

instead of tab you could use , or \| as the default tokenization is done using a java Regex pattern {code}[,\t](,\t.html)
*[,|\t][ ,\t]*{code}
You can override this parameter to parse your log files or transaction
files (each line is a transaction.) The FPGrowth algorithm mines the top K
frequently occurring sets of items and their counts from the given input
data

$MAHOUT_HOME/core/src/test/resources/retail.dat is a sample dataset in this
format. 
Other sample files are accident.dat.gz from [http://fimi.cs.helsinki.fi/data/](http://fimi.cs.helsinki.fi/data/)
. As a quick test, try this:


    bin/mahout fpg \
         -i core/src/test/resources/retail.dat \
         -o patterns \
         -k 50 \
         -method sequential \
         -regex '[\ ]
' \
         -s 2


The minimumSupport parameter \-s is the minimum number of times a pattern
or a feature needs to occur in the dataset so that it is included in the
patterns generated. You can speed up the process by having a large value of
s. There are cases where you will have less than k patterns for a
particular feature as the rest don't for qualify the minimum support
criteria

Note that the input to the algorithm, could be uncompressed or compressed
gz file or even a directory containing any number of such files.
We modified the regex to use space to split the token. Note that input
regex string is escaped.

<a name="ParallelFrequentPatternMining-RunningParallelFPGrowth"></a>
## Running Parallel FPGrowth

Running parallel FPGrowth is as easy as adding changing the flag \-method
mapreduce and adding the number of groups parameter e.g. \-g 20 for 20
groups. First, let's run the above sample test in map-reduce mode:

    bin/mahout fpg \
         -i core/src/test/resources/retail.dat \
         -o patterns \
         -k 50 \
         -method mapreduce \
         -regex '[\ ]
' \
         -s 2

The above test took 102 seconds on dual-core laptop, v.s. 609 seconds in
the sequential mode, (with 5 gigs of ram allocated). In a separate test,
the first 1000 lines of retail.dat took 20 seconds in map/reduce v.s. 30
seconds in sequential mode.

Here is another dataset which, while several times larger, requires much
less time to find frequent patterns, as there are very few. Get
accidents.dat.gz from [http://fimi.cs.helsinki.fi/data/](http://fimi.cs.helsinki.fi/data/)
 and place it on your hdfs in a folder named accidents. Then, run the
hadoop version of the FPGrowth job:

    bin/mahout fpg \
         -i accidents \
         -o patterns \
         -k 50 \
         -method mapreduce \
         -regex '[\ ]
' \
         -s 2


OR to run a dataset of this size in sequential mode on a single machine
let's give Mahout a lot more memory and only keep features with more than
300 members:

    export MAHOUT_HEAPSIZE=-Xmx5000m
    bin/mahout fpg \
         -i accidents \
         -o patterns \
         -k 50 \
         -method sequential \
         -regex '[\ ]
' \
         -s 2



The numGroups parameter \-g in FPGrowthJob specifies the number of groups
into which transactions have to be decomposed. The default of 1000 works
very well on a single-machine cluster; this may be very different on large
clusters.

Note that accidents.dat has 340 unique features. So we chose \-g 10 to
split the transactions across 10 shards where 34 patterns are mined from
each shard. (Note: g doesnt need to be exactly divisible.) The Algorithm
takes care of calculating the split. For better performance in large
datasets and clusters, try not to mine for more than 20-25 features per
shard. Stick to the defaults on a small machine.

The numTreeCacheEntries parameter \-tc specifies the number of generated
conditional FP-Trees to be kept in memory so that subsequent operations do
not to regenerate them. Increasing this number increases the memory
consumption but might improve speed until a certain point. This depends
entirely on the dataset in question. A value of 5-10 is recommended for
mining up to top 100 patterns for each feature.

<a name="ParallelFrequentPatternMining-Viewingtheresults"></a>
## Viewing the results
The output will be dumped to a SequenceFile in the frequentpatterns
directory in Text=>TopKStringPatterns format. Run this command to see a few
of the Frequent Patterns:

    bin/mahout seqdumper \
         -i patterns/frequentpatterns/part-?-00000 \
         -n 4

or replace -n 4 with -c for the count of patterns.
 
Open questions: how does one experiment and monitor with these various
parameters?
