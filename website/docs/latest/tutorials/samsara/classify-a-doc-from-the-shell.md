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
title: Text Classification Example

    
---

# Building a text classifier in Mahout's Spark Shell

This tutorial will take you through the steps used to train a Multinomial Naive Bayes model and create a text classifier based on that model using the ```mahout spark-shell```. 

## Prerequisites
This tutorial assumes that you have your Spark environment variables set for the ```mahout spark-shell``` see: [Playing with Mahout's Shell](http://mahout.apache.org/users/sparkbindings/play-with-shell.html).  As well we assume that Mahout is running in cluster mode (i.e. with the ```MAHOUT_LOCAL``` environment variable **unset**) as we'll be reading and writing to HDFS.

## Downloading and Vectorizing the Wikipedia dataset
*As of Mahout v. 0.10.0, we are still reliant on the MapReduce versions of ```mahout seqwiki``` and ```mahout seq2sparse``` to extract and vectorize our text.  A* [*Spark implementation of seq2sparse*](https://issues.apache.org/jira/browse/MAHOUT-1663) *is in the works for Mahout v. 0.11.* However, to download the Wikipedia dataset, extract the bodies of the documentation, label each document and vectorize the text into TF-IDF vectors, we can simpmly run the [wikipedia-classifier.sh](https://github.com/apache/mahout/blob/master/examples/bin/classify-wikipedia.sh) example.  

    Please select a number to choose the corresponding task to run
    1. CBayes (may require increased heap space on yarn)
    2. BinaryCBayes
    3. clean -- cleans up the work area in /tmp/mahout-work-wiki
    Enter your choice :

Enter (2). This will download a large recent XML dump of the Wikipedia database, into a ```/tmp/mahout-work-wiki``` directory, unzip it and  place it into HDFS.  It will run a [MapReduce job to parse the wikipedia set](http://mahout.apache.org/users/classification/wikipedia-classifier-example.html), extracting and labeling only pages with category tags for [United States] and [United Kingdom] (~11600 documents). It will then run ```mahout seq2sparse``` to convert the documents into TF-IDF vectors.  The script will also a build and test a [Naive Bayes model using MapReduce](http://mahout.apache.org/users/classification/bayesian.html).  When it is completed, you should see a confusion matrix on your screen.  For this tutorial, we will ignore the MapReduce model, and build a new model using Spark based on the vectorized text output by ```seq2sparse```.

## Getting Started

Launch the ```mahout spark-shell```.  There is an example script: ```spark-document-classifier.mscala``` (.mscala denotes a Mahout-Scala script which can be run similarly to an R script).   We will be walking through this script for this tutorial but if you wanted to simply run the script, you could just issue the command: 

    mahout> :load /path/to/mahout/examples/bin/spark-document-classifier.mscala

For now, lets take the script apart piece by piece.  You can cut and paste the following code blocks into the ```mahout spark-shell```.

## Imports

Our Mahout Naive Bayes imports:

    import org.apache.mahout.classifier.naivebayes._
    import org.apache.mahout.classifier.stats._
    import org.apache.mahout.nlp.tfidf._

Hadoop imports needed to read our dictionary:

    import org.apache.hadoop.io.Text
    import org.apache.hadoop.io.IntWritable
    import org.apache.hadoop.io.LongWritable

## Read in our full set from HDFS as vectorized by seq2sparse in classify-wikipedia.sh

    val pathToData = "/tmp/mahout-work-wiki/"
    val fullData = drmDfsRead(pathToData + "wikipediaVecs/tfidf-vectors")

## Extract the category of each observation and aggregate those observations by category

    val (labelIndex, aggregatedObservations) = SparkNaiveBayes.extractLabelsAndAggregateObservations(
                                                                 fullData)

## Build a Muitinomial Naive Bayes model and self test on the training set

    val model = SparkNaiveBayes.train(aggregatedObservations, labelIndex, false)
    val resAnalyzer = SparkNaiveBayes.test(model, fullData, false)
    println(resAnalyzer)
    
printing the ```ResultAnalyzer``` will display the confusion matrix.

## Read in the dictionary and document frequency count from HDFS
    
    val dictionary = sdc.sequenceFile(pathToData + "wikipediaVecs/dictionary.file-0",
                                      classOf[Text],
                                      classOf[IntWritable])
    val documentFrequencyCount = sdc.sequenceFile(pathToData + "wikipediaVecs/df-count",
                                                  classOf[IntWritable],
                                                  classOf[LongWritable])

    // setup the dictionary and document frequency count as maps
    val dictionaryRDD = dictionary.map { 
                                    case (wKey, wVal) => wKey.asInstanceOf[Text]
                                                             .toString() -> wVal.get() 
                                       }
                                       
    val documentFrequencyCountRDD = documentFrequencyCount.map {
                                            case (wKey, wVal) => wKey.asInstanceOf[IntWritable]
                                                                     .get() -> wVal.get() 
                                                               }
    
    val dictionaryMap = dictionaryRDD.collect.map(x => x._1.toString -> x._2.toInt).toMap
    val dfCountMap = documentFrequencyCountRDD.collect.map(x => x._1.toInt -> x._2.toLong).toMap

## Define a function to tokenize and vectorize new text using our current dictionary

For this simple example, our function ```vectorizeDocument(...)``` will tokenize a new document into unigrams using native Java String methods and vectorize using our dictionary and document frequencies. You could also use a [Lucene](https://lucene.apache.org/core/) analyzer for bigrams, trigrams, etc., and integrate Apache [Tika](https://tika.apache.org/) to extract text from different document types (PDF, PPT, XLS, etc.).  Here, however we will keep it simple, stripping and tokenizing our text using regexs and native String methods.

    def vectorizeDocument(document: String,
                            dictionaryMap: Map[String,Int],
                            dfMap: Map[Int,Long]): Vector = {
        val wordCounts = document.replaceAll("[^\\p{L}\\p{Nd}]+", " ")
                                    .toLowerCase
                                    .split(" ")
                                    .groupBy(identity)
                                    .mapValues(_.length)         
        val vec = new RandomAccessSparseVector(dictionaryMap.size)
        val totalDFSize = dfMap(-1)
        val docSize = wordCounts.size
        for (word <- wordCounts) {
            val term = word._1
            if (dictionaryMap.contains(term)) {
                val tfidf: TermWeight = new TFIDF()
                val termFreq = word._2
                val dictIndex = dictionaryMap(term)
                val docFreq = dfCountMap(dictIndex)
                val currentTfIdf = tfidf.calculate(termFreq,
                                                   docFreq.toInt,
                                                   docSize,
                                                   totalDFSize.toInt)
                vec.setQuick(dictIndex, currentTfIdf)
            }
        }
        vec
    }

## Setup our classifier

    val labelMap = model.labelIndex
    val numLabels = model.numLabels
    val reverseLabelMap = labelMap.map(x => x._2 -> x._1)
    
    // instantiate the correct type of classifier
    val classifier = model.isComplementary match {
        case true => new ComplementaryNBClassifier(model)
        case _ => new StandardNBClassifier(model)
    }

## Define an argmax function 

The label with the highest score wins the classification for a given document.
    
    def argmax(v: Vector): (Int, Double) = {
        var bestIdx: Int = Integer.MIN_VALUE
        var bestScore: Double = Integer.MIN_VALUE.asInstanceOf[Int].toDouble
        for(i <- 0 until v.size) {
            if(v(i) > bestScore){
                bestScore = v(i)
                bestIdx = i
            }
        }
        (bestIdx, bestScore)
    }

## Define our TF(-IDF) vector classifier

    def classifyDocument(clvec: Vector) : String = {
        val cvec = classifier.classifyFull(clvec)
        val (bestIdx, bestScore) = argmax(cvec)
        reverseLabelMap(bestIdx)
    }

## Two sample news articles: United States Football and United Kingdom Football
    
    // A random United States football article
    // http://www.reuters.com/article/2015/01/28/us-nfl-superbowl-security-idUSKBN0L12JR20150128
    val UStextToClassify = new String("(Reuters) - Super Bowl security officials acknowledge" +
        " the NFL championship game represents a high profile target on a world stage but are" +
        " unaware of any specific credible threats against Sunday's showcase. In advance of" +
        " one of the world's biggest single day sporting events, Homeland Security Secretary" +
        " Jeh Johnson was in Glendale on Wednesday to review security preparations and tour" +
        " University of Phoenix Stadium where the Seattle Seahawks and New England Patriots" +
        " will battle. Deadly shootings in Paris and arrest of suspects in Belgium, Greece and" +
        " Germany heightened fears of more attacks around the world and social media accounts" +
        " linked to Middle East militant groups have carried a number of threats to attack" +
        " high-profile U.S. events. There is no specific credible threat, said Johnson, who" + 
        " has appointed a federal coordination team to work with local, state and federal" +
        " agencies to ensure safety of fans, players and other workers associated with the" + 
        " Super Bowl. I'm confident we will have a safe and secure and successful event." +
        " Sunday's game has been given a Special Event Assessment Rating (SEAR) 1 rating, the" +
        " same as in previous years, except for the year after the Sept. 11, 2001 attacks, when" +
        " a higher level was declared. But security will be tight and visible around Super" +
        " Bowl-related events as well as during the game itself. All fans will pass through" +
        " metal detectors and pat downs. Over 4,000 private security personnel will be deployed" +
        " and the almost 3,000 member Phoenix police force will be on Super Bowl duty. Nuclear" +
        " device sniffing teams will be deployed and a network of Bio-Watch detectors will be" +
        " set up to provide a warning in the event of a biological attack. The Department of" +
        " Homeland Security (DHS) said in a press release it had held special cyber-security" +
        " and anti-sniper training sessions. A U.S. official said the Transportation Security" +
        " Administration, which is responsible for screening airline passengers, will add" +
        " screeners and checkpoint lanes at airports. Federal air marshals, behavior detection" +
        " officers and dog teams will help to secure transportation systems in the area. We" +
        " will be ramping it (security) up on Sunday, there is no doubt about that, said Federal"+
        " Coordinator Matthew Allen, the DHS point of contact for planning and support. I have" +
        " every confidence the public safety agencies that represented in the planning process" +
        " are going to have their best and brightest out there this weekend and we will have" +
        " a very safe Super Bowl.")
    
    // A random United Kingdom football article
    // http://www.reuters.com/article/2015/01/26/manchester-united-swissquote-idUSL6N0V52RZ20150126
    val UKtextToClassify = new String("(Reuters) - Manchester United have signed a sponsorship" +
        " deal with online financial trading company Swissquote, expanding the commercial" +
        " partnerships that have helped to make the English club one of the richest teams in" +
        " world soccer. United did not give a value for the deal, the club's first in the sector," +
        " but said on Monday it was a multi-year agreement. The Premier League club, 20 times" +
        " English champions, claim to have 659 million followers around the globe, making the" +
        " United name attractive to major brands like Chevrolet cars and sportswear group Adidas." +
        " Swissquote said the global deal would allow it to use United's popularity in Asia to" +
        " help it meet its targets for expansion in China. Among benefits from the deal," +
        " Swissquote's clients will have a chance to meet United players and get behind the scenes" +
        " at the Old Trafford stadium. Swissquote is a Geneva-based online trading company that" +
        " allows retail investors to buy and sell foreign exchange, equities, bonds and other asset" +
        " classes. Like other retail FX brokers, Swissquote was left nursing losses on the Swiss" +
        " franc after Switzerland's central bank stunned markets this month by abandoning its cap" +
        " on the currency. The fallout from the abrupt move put rival and West Ham United shirt" +
        " sponsor Alpari UK into administration. Swissquote itself was forced to book a 25 million" +
        " Swiss francs ($28 million) provision for its clients who were left out of pocket" +
        " following the franc's surge. United's ability to grow revenues off the pitch has made" +
        " them the second richest club in the world behind Spain's Real Madrid, despite a" +
        " downturn in their playing fortunes. United Managing Director Richard Arnold said" +
        " there was still lots of scope for United to develop sponsorships in other areas of" +
        " business. The last quoted statistics that we had showed that of the top 25 sponsorship" +
        " categories, we were only active in 15 of those, Arnold told Reuters. I think there is a" +
        " huge potential still for the club, and the other thing we have seen is there is very" +
        " significant growth even within categories. United have endured a tricky transition" +
        " following the retirement of manager Alex Ferguson in 2013, finishing seventh in the" +
        " Premier League last season and missing out on a place in the lucrative Champions League." +
        " ($1 = 0.8910 Swiss francs) (Writing by Neil Maidment, additional reporting by Jemima" + 
        " Kelly; editing by Keith Weir)")

## Vectorize and classify our documents

    val usVec = vectorizeDocument(UStextToClassify, dictionaryMap, dfCountMap)
    val ukVec = vectorizeDocument(UKtextToClassify, dictionaryMap, dfCountMap)
    
    println("Classifying the news article about superbowl security (united states)")
    classifyDocument(usVec)
    
    println("Classifying the news article about Manchester United (united kingdom)")
    classifyDocument(ukVec)

## Tie everything together in a new method to classify text 
    
    def classifyText(txt: String): String = {
        val v = vectorizeDocument(txt, dictionaryMap, dfCountMap)
        classifyDocument(v)
    }

## Now we can simply call our classifyText(...) method on any String

    classifyText("Hello world from Queens")
    classifyText("Hello world from London")
    
## Model persistance

You can save the model to HDFS:

    model.dfsWrite("/path/to/model")
    
And retrieve it with:

    val model =  NBModel.dfsRead("/path/to/model")

The trained model can now be embedded in an external application.