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
title: (Deprecated)  Perceptron and Winnow

    
---

#Intro to Cooccurrence Recommenders with Spark

Mahout provides several important building blocks for creating recommendations using Spark. *spark-itemsimilarity* can 
be used to create "other people also liked these things" type recommendations and paired with a search engine can 
personalize recommendations for individual users. *spark-rowsimilarity* can provide non-personalized content based 
recommendations and when paired with a search engine can be used to personalize content based recommendations.

##References

1. A free ebook, which talks about the general idea: [Practical Machine Learning](https://www.mapr.com/practical-machine-learning)
2. A slide deck, which talks about mixing actions or other indicators: [Creating a Unified Recommender](http://occamsmachete.com/ml/2014/10/07/creating-a-unified-recommender-with-mahout-and-a-search-engine/)
3. Two blog posts: [What's New in Recommenders: part #1](http://occamsmachete.com/ml/2014/08/11/mahout-on-spark-whats-new-in-recommenders/)
and  [What's New in Recommenders: part #2](http://occamsmachete.com/ml/2014/09/09/mahout-on-spark-whats-new-in-recommenders-part-2/)
3. A post describing the loglikelihood ratio:  [Surprise and Coinsidense](http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html)  LLR is used to reduce noise in the data while keeping the calculations O(n) complexity.

Below are the command line jobs but the drivers and associated code can also be customized and accessed from the Scala APIs.

##1. spark-itemsimilarity
*spark-itemsimilarity* is the Spark counterpart of the of the Mahout mapreduce job called *itemsimilarity*. It takes in elements of interactions, which have userID, itemID, and optionally a value. It will produce one of more indicator matrices created by comparing every user's interactions with every other user. The indicator matrix is an item x item matrix where the values are log-likelihood ratio strengths. For the legacy mapreduce version, there were several possible similarity measures but these are being deprecated in favor of LLR because in practice it performs the best.

Mahout's mapreduce version of itemsimilarity takes a text file that is expected to have user and item IDs that conform to 
Mahout's ID requirements--they are non-negative integers that can be viewed as row and column numbers in a matrix.

*spark-itemsimilarity* also extends the notion of cooccurrence to cross-cooccurrence, in other words the Spark version will 
account for multi-modal interactions and create indicator matrices allowing the use of much more data in 
creating recommendations or similar item lists. People try to do this by mixing different actions and giving them weights. 
For instance they might say an item-view is 0.2 of an item purchase. In practice this is often not helpful. Spark-itemsimilarity's
cross-cooccurrence is a more principled way to handle this case. In effect it scrubs secondary actions with the action you want
to recommend.   


    spark-itemsimilarity Mahout 1.0
    Usage: spark-itemsimilarity [options]
    
    Disconnected from the target VM, address: '127.0.0.1:64676', transport: 'socket'
    Input, output options
      -i <value> | --input <value>
            Input path, may be a filename, directory name, or comma delimited list of HDFS supported URIs (required)
      -i2 <value> | --input2 <value>
            Secondary input path for cross-similarity calculation, same restrictions as "--input" (optional). Default: empty.
      -o <value> | --output <value>
            Path for output, any local or HDFS supported URI (required)
    
    Algorithm control options:
      -mppu <value> | --maxPrefs <value>
            Max number of preferences to consider per user (optional). Default: 500
      -m <value> | --maxSimilaritiesPerItem <value>
            Limit the number of similarities per item to this number (optional). Default: 100
    
    Note: Only the Log Likelihood Ratio (LLR) is supported as a similarity measure.
    
    Input text file schema options:
      -id <value> | --inDelim <value>
            Input delimiter character (optional). Default: "[,\t]"
      -f1 <value> | --filter1 <value>
            String (or regex) whose presence indicates a datum for the primary item set (optional). Default: no filter, all data is used
      -f2 <value> | --filter2 <value>
            String (or regex) whose presence indicates a datum for the secondary item set (optional). If not present no secondary dataset is collected
      -rc <value> | --rowIDColumn <value>
            Column number (0 based Int) containing the row ID string (optional). Default: 0
      -ic <value> | --itemIDColumn <value>
            Column number (0 based Int) containing the item ID string (optional). Default: 1
      -fc <value> | --filterColumn <value>
            Column number (0 based Int) containing the filter string (optional). Default: -1 for no filter
    
    Using all defaults the input is expected of the form: "userID<tab>itemId" or "userID<tab>itemID<tab>any-text..." and all rows will be used
    
    File discovery options:
      -r | --recursive
            Searched the -i path recursively for files that match --filenamePattern (optional), Default: false
      -fp <value> | --filenamePattern <value>
            Regex to match in determining input files (optional). Default: filename in the --input option or "^part-.*" if --input is a directory
    
    Output text file schema options:
      -rd <value> | --rowKeyDelim <value>
            Separates the rowID key from the vector values list (optional). Default: "\t"
      -cd <value> | --columnIdStrengthDelim <value>
            Separates column IDs from their values in the vector values list (optional). Default: ":"
      -td <value> | --elementDelim <value>
            Separates vector element values in the values list (optional). Default: " "
      -os | --omitStrength
            Do not write the strength to the output files (optional), Default: false.
    This option is used to output indexable data for creating a search engine recommender.
    
    Default delimiters will produce output of the form: "itemID1<tab>itemID2:value2<space>itemID10:value10..."
    
    Spark config options:
      -ma <value> | --master <value>
            Spark Master URL (optional). Default: "local". Note that you can specify the number of cores to get a performance improvement, for example "local[4]"
      -sem <value> | --sparkExecutorMem <value>
            Max Java heap available as "executor memory" on each node (optional). Default: 4g
      -rs <value> | --randomSeed <value>
            
      -h | --help
            prints this usage text

This looks daunting but defaults to simple fairly sane values to take exactly the same input as legacy code and is pretty flexible. It allows the user to point to a single text file, a directory full of files, or a tree of directories to be traversed recursively. The files included can be specified with either a regex-style pattern or filename. The schema for the file is defined by column numbers, which map to the important bits of data including IDs and values. The files can even contain filters, which allow unneeded rows to be discarded or used for cross-cooccurrence calculations.

See ItemSimilarityDriver.scala in Mahout's spark module if you want to customize the code. 

###Defaults in the _**spark-itemsimilarity**_ CLI

If all defaults are used the input can be as simple as:

    userID1,itemID1
    userID2,itemID2
    ...

With the command line:


    bash$ mahout spark-itemsimilarity --input in-file --output out-dir


This will use the "local" Spark context and will output the standard text version of a DRM

    itemID1<tab>itemID2:value2<space>itemID10:value10...

###<a name="multiple-actions">How To Use Multiple User Actions</a>

Often we record various actions the user takes for later analytics. These can now be used to make recommendations. 
The idea of a recommender is to recommend the action you want the user to make. For an ecom app this might be 
a purchase action. It is usually not a good idea to just treat other actions the same as the action you want to recommend. 
For instance a view of an item does not indicate the same intent as a purchase and if you just mixed the two together you 
might even make worse recommendations. It is tempting though since there are so many more views than purchases. With *spark-itemsimilarity*
we can now use both actions. Mahout will use cross-action cooccurrence analysis to limit the views to ones that do predict purchases.
We do this by treating the primary action (purchase) as data for the indicator matrix and use the secondary action (view) 
to calculate the cross-cooccurrence indicator matrix.  

*spark-itemsimilarity* can read separate actions from separate files or from a mixed action log by filtering certain lines. For a mixed 
action log of the form:

    u1,purchase,iphone
    u1,purchase,ipad
    u2,purchase,nexus
    u2,purchase,galaxy
    u3,purchase,surface
    u4,purchase,iphone
    u4,purchase,galaxy
    u1,view,iphone
    u1,view,ipad
    u1,view,nexus
    u1,view,galaxy
    u2,view,iphone
    u2,view,ipad
    u2,view,nexus
    u2,view,galaxy
    u3,view,surface
    u3,view,nexus
    u4,view,iphone
    u4,view,ipad
    u4,view,galaxy

###Command Line


Use the following options:

    bash$ mahout spark-itemsimilarity \
    	--input in-file \     # where to look for data
        --output out-path \   # root dir for output
        --master masterUrl \  # URL of the Spark master server
        --filter1 purchase \  # word that flags input for the primary action
        --filter2 view \      # word that flags input for the secondary action
        --itemIDPosition 2 \  # column that has the item ID
        --rowIDPosition 0 \   # column that has the user ID
        --filterPosition 1    # column that has the filter word



###Output

The output of the job will be the standard text version of two Mahout DRMs. This is a case where we are calculating 
cross-cooccurrence so a primary indicator matrix and cross-cooccurrence indicator matrix will be created

    out-path
      |-- similarity-matrix - TDF part files
      \-- cross-similarity-matrix - TDF part-files

The indicator matrix will contain the lines:

    galaxy\tnexus:1.7260924347106847
    ipad\tiphone:1.7260924347106847
    nexus\tgalaxy:1.7260924347106847
    iphone\tipad:1.7260924347106847
    surface

The cross-cooccurrence indicator matrix will contain:

    iphone\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847
    ipad\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897
    nexus\tnexus:0.6795961471815897 iphone:0.6795961471815897 ipad:0.6795961471815897 galaxy:0.6795961471815897
    galaxy\tnexus:1.7260924347106847 iphone:1.7260924347106847 ipad:1.7260924347106847 galaxy:1.7260924347106847
    surface\tsurface:4.498681156950466 nexus:0.6795961471815897

**Note:** You can run this multiple times to use more than two actions or you can use the underlying 
SimilarityAnalysis.cooccurrence API, which will more efficiently calculate any number of cross-cooccurrence indicators.

###Log File Input

A common method of storing data is in log files. If they are written using some delimiter they can be consumed directly by spark-itemsimilarity. For instance input of the form:

    2014-06-23 14:46:53.115\tu1\tpurchase\trandom text\tiphone
    2014-06-23 14:46:53.115\tu1\tpurchase\trandom text\tipad
    2014-06-23 14:46:53.115\tu2\tpurchase\trandom text\tnexus
    2014-06-23 14:46:53.115\tu2\tpurchase\trandom text\tgalaxy
    2014-06-23 14:46:53.115\tu3\tpurchase\trandom text\tsurface
    2014-06-23 14:46:53.115\tu4\tpurchase\trandom text\tiphone
    2014-06-23 14:46:53.115\tu4\tpurchase\trandom text\tgalaxy
    2014-06-23 14:46:53.115\tu1\tview\trandom text\tiphone
    2014-06-23 14:46:53.115\tu1\tview\trandom text\tipad
    2014-06-23 14:46:53.115\tu1\tview\trandom text\tnexus
    2014-06-23 14:46:53.115\tu1\tview\trandom text\tgalaxy
    2014-06-23 14:46:53.115\tu2\tview\trandom text\tiphone
    2014-06-23 14:46:53.115\tu2\tview\trandom text\tipad
    2014-06-23 14:46:53.115\tu2\tview\trandom text\tnexus
    2014-06-23 14:46:53.115\tu2\tview\trandom text\tgalaxy
    2014-06-23 14:46:53.115\tu3\tview\trandom text\tsurface
    2014-06-23 14:46:53.115\tu3\tview\trandom text\tnexus
    2014-06-23 14:46:53.115\tu4\tview\trandom text\tiphone
    2014-06-23 14:46:53.115\tu4\tview\trandom text\tipad
    2014-06-23 14:46:53.115\tu4\tview\trandom text\tgalaxy    

Can be parsed with the following CLI and run on the cluster producing the same output as the above example.

    bash$ mahout spark-itemsimilarity \
        --input in-file \
        --output out-path \
        --master spark://sparkmaster:4044 \
        --filter1 purchase \
        --filter2 view \
        --inDelim "\t" \
        --itemIDPosition 4 \
        --rowIDPosition 1 \
        --filterPosition 2

##2. spark-rowsimilarity

*spark-rowsimilarity* is the companion to *spark-itemsimilarity* the primary difference is that it takes a text file version of 
a matrix of sparse vectors with optional application specific IDs and it finds similar rows rather than items (columns). Its use is
not limited to collaborative filtering. The input is in text-delimited form where there are three delimiters used. By 
default it reads (rowID&lt;tab>columnID1:strength1&lt;space>columnID2:strength2...) Since this job only supports LLR similarity,
 which does not use the input strengths, they may be omitted in the input. It writes 
(rowID&lt;tab>rowID1:strength1&lt;space>rowID2:strength2...) 
The output is sorted by strength descending. The output can be interpreted as a row ID from the primary input followed 
by a list of the most similar rows.

The command line interface is:

    spark-rowsimilarity Mahout 1.0
    Usage: spark-rowsimilarity [options]
    
    Input, output options
      -i <value> | --input <value>
            Input path, may be a filename, directory name, or comma delimited list of HDFS supported URIs (required)
      -o <value> | --output <value>
            Path for output, any local or HDFS supported URI (required)
    
    Algorithm control options:
      -mo <value> | --maxObservations <value>
            Max number of observations to consider per row (optional). Default: 500
      -m <value> | --maxSimilaritiesPerRow <value>
            Limit the number of similarities per item to this number (optional). Default: 100
    
    Note: Only the Log Likelihood Ratio (LLR) is supported as a similarity measure.
    Disconnected from the target VM, address: '127.0.0.1:49162', transport: 'socket'
    
    Output text file schema options:
      -rd <value> | --rowKeyDelim <value>
            Separates the rowID key from the vector values list (optional). Default: "\t"
      -cd <value> | --columnIdStrengthDelim <value>
            Separates column IDs from their values in the vector values list (optional). Default: ":"
      -td <value> | --elementDelim <value>
            Separates vector element values in the values list (optional). Default: " "
      -os | --omitStrength
            Do not write the strength to the output files (optional), Default: false.
    This option is used to output indexable data for creating a search engine recommender.
    
    Default delimiters will produce output of the form: "itemID1<tab>itemID2:value2<space>itemID10:value10..."
    
    File discovery options:
      -r | --recursive
            Searched the -i path recursively for files that match --filenamePattern (optional), Default: false
      -fp <value> | --filenamePattern <value>
            Regex to match in determining input files (optional). Default: filename in the --input option or "^part-.*" if --input is a directory
    
    Spark config options:
      -ma <value> | --master <value>
            Spark Master URL (optional). Default: "local". Note that you can specify the number of cores to get a performance improvement, for example "local[4]"
      -sem <value> | --sparkExecutorMem <value>
            Max Java heap available as "executor memory" on each node (optional). Default: 4g
      -rs <value> | --randomSeed <value>
            
      -h | --help
            prints this usage text

See RowSimilarityDriver.scala in Mahout's spark module if you want to customize the code. 

#3. Using *spark-rowsimilarity* with Text Data

Another use case for *spark-rowsimilarity* is in finding similar textual content. For instance given the tags associated with 
a blog post,
 which other posts have similar tags. In this case the columns are tags and the rows are posts. Since LLR is 
the only similarity method supported this is not the optimal way to determine general "bag-of-words" document similarity. 
LLR is used more as a quality filter than as a similarity measure. However *spark-rowsimilarity* will produce 
lists of similar docs for every doc if input is docs with lists of terms. The Apache [Lucene](http://lucene.apache.org) project provides several methods of [analyzing and tokenizing](http://lucene.apache.org/core/4_9_0/core/org/apache/lucene/analysis/package-summary.html#package_description) documents.

#<a name="unified-recommender">4. Creating a Unified Recommender</a>

Using the output of *spark-itemsimilarity* and *spark-rowsimilarity* you can build a unified cooccurrence and content based
 recommender that can be used in both or either mode depending on indicators available and the history available at 
runtime for a user.

##Requirements

1. Mahout 0.10.0 or later
2. Hadoop
3. Spark, the correct version for your version of Mahout and Hadoop
4. A search engine like Solr or Elasticsearch

##Indicators

Indicators come in 3 types

1. **Cooccurrence**: calculated with *spark-itemsimilarity* from user actions
2. **Content**: calculated from item metadata or content using *spark-rowsimilarity*
3. **Intrinsic**: assigned to items as metadata. Can be anything that describes the item.

The query for recommendations will be a mix of values meant to match one of your indicators. The query can be constructed 
from user history and values derived from context (category being viewed for instance) or special precalculated data 
(popularity rank for instance). This blending of indicators allows for creating many flavors or recommendations to fit 
a very wide variety of circumstances.

With the right mix of indicators developers can construct a single query that works for completely new items and new users 
while working well for items with lots of interactions and users with many recorded actions. In other words by adding in content and intrinsic 
indicators developers can create a solution for the "cold-start" problem that gracefully improves with more user history
and as items have more interactions. It is also possible to create a completely content-based recommender that personalizes 
recommendations.

##Example with 3 Indicators

You will need to decide how you store user action data so they can be processed by the item and row similarity jobs and 
this is most easily done by using text files as described above. The data that is processed by these jobs is considered the 
training data. You will need some amount of user history in your recs query. It is typical to use the most recent user history 
but need not be exactly what is in the training set, which may include a greater volume of historical data. Keeping the user 
history for query purposes could be done with a database by storing it in a users table. In the example above the two 
collaborative filtering actions are "purchase" and "view", but let's also add tags (taken from catalog categories or other 
descriptive metadata). 

We will need to create 1 cooccurrence indicator from the primary action (purchase) 1 cross-action cooccurrence indicator 
from the secondary action (view) 
and 1 content indicator (tags). We'll have to run *spark-itemsimilarity* once and *spark-rowsimilarity* once.

We have described how to create the collaborative filtering indicator and cross-cooccurrence indicator for purchase and view (the [How to use Multiple User 
Actions](#multiple-actions) section) but tags will be a slightly different process. We want to use the fact that 
certain items have tags similar to the ones associated with a user's purchases. This is not a collaborative filtering indicator 
but rather a "content" or "metadata" type indicator since you are not using other users' history, only the 
individual that you are making recs for. This means that this method will make recommendations for items that have 
no collaborative filtering data, as happens with new items in a catalog. New items may have tags assigned but no one
 has purchased or viewed them yet. In the final query we will mix all 3 indicators.

##Content Indicator

To create a content-indicator we'll make use of the fact that the user has purchased items with certain tags. We want to find 
items with the most similar tags. Notice that other users' behavior is not considered--only other item's tags. This defines a 
content or metadata indicator. They are used when you want to find items that are similar to other items by using their 
content or metadata, not by which users interacted with them.

For this we need input of the form:

    itemID<tab>list-of-tags
    ...

The full collection will look like the tags column from a catalog DB. For our ecom example it might be:

    3459860b<tab>men long-sleeve chambray clothing casual
    9446577d<tab>women tops chambray clothing casual
    ...

We'll use *spark-rowimilairity* because we are looking for similar rows, which encode items in this case. As with the 
collaborative filtering indicator and cross-cooccurrence indicator we use the --omitStrength option. The strengths created are 
probabilistic log-likelihood ratios and so are used to filter unimportant similarities. Once the filtering or downsampling 
is finished we no longer need the strengths. We will get an indicator matrix of the form:

    itemID<tab>list-of-item IDs
    ...

This is a content indicator since it has found other items with similar content or metadata.

    3459860b<tab>3459860b 3459860b 6749860c 5959860a 3434860a 3477860a
    9446577d<tab>9446577d 9496577d 0943577d 8346577d 9442277d 9446577e
    ...  
    
We now have three indicators, two collaborative filtering type and one content type.

##Unified Recommender Query

The actual form of the query for recommendations will vary depending on your search engine but the intent is the same. 
For a given user, map their history of an action or content to the correct indicator field and perform an OR'd query. 

We have 3 indicators, these are indexed by the search engine into 3 fields, we'll call them "purchase", "view", and "tags". 
We take the user's history that corresponds to each indicator and create a query of the form:

    Query:
      field: purchase; q:user's-purchase-history
      field: view; q:user's view-history
      field: tags; q:user's-tags-associated-with-purchases
      
The query will result in an ordered list of items recommended for purchase but skewed towards items with similar tags to 
the ones the user has already purchased. 

This is only an example and not necessarily the optimal way to create recs. It illustrates how business decisions can be 
translated into recommendations. This technique can be used to skew recommendations towards intrinsic indicators also. 
For instance you may want to put personalized popular item recs in a special place in the UI. Create a popularity indicator 
by tagging items with some category of popularity (hot, warm, cold for instance) then
index that as a new indicator field and include the corresponding value in a query 
on the popularity field. If we use the ecom example but use the query to get "hot" recommendations it might look like this:

    Query:
      field: purchase; q:user's-purchase-history
      field: view; q:user's view-history
      field: popularity; q:"hot"

This will return recommendations favoring ones that have the intrinsic indicator "hot".

##Notes
1. Use as much user action history as you can gather. Choose a primary action that is closest to what you want to recommend and the others will be used to create cross-cooccurrence indicators. Using more data in this fashion will almost always produce better recommendations.
2. Content can be used where there is no recorded user behavior or when items change too quickly to get much interaction history. They can be used alone or mixed with other indicators.
3. Most search engines support "boost" factors so you can favor one or more indicators. In the example query, if you want tags to only have a small effect you could boost the CF indicators.
4. In the examples we have used space delimited strings for lists of IDs in indicators and in queries. It may be better to use arrays of strings if your storage system and search engine support them. For instance Solr allows multi-valued fields, which correspond to arrays.
