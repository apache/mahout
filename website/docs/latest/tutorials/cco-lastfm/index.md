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
title: CCOs with Last.fm

    
---

Most reccomender examples utilize the MovieLense dataset, but that relies only on ratings (which makes the recommender being demonstrated look less trivial).  Right next to the MovieLense dataset is the LastFM data set.  The LastFM dataset has ratings by user, friends of the user, bands listened to by user, and tags by user.  This is the kind of exciting data set we’d like to work with!

Start by downloading the LastFM dataset from 
http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

I’m going to assume you’ve unzipped them to /path/to/lastfm/*
We’re going to use a new trick for creating our IndexedDataSets, the `apply` function.  `apply` takes an `RDD[(String, String)]` that is an RDD of tuples where both elements are strings. We load RDDs, and use Spark to manipulate the RDDs into this form.  The files from LastFM are tab seperated- but it should be noted, that this could easily be done from log files, but would just take a touch more Spark-Fu.  

The second important thing to note is that the first element in each tuple is going to be the rows in the resulting matrix, the second element will be the column, and at that position there will be a one.  The BiDictionary will automatically be created from the strings. 
For those following along at home- [the full Scala worksheet](cco-lastfm.scala) might be easier than copying and pasting 
from this page.

~~~
import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark

val userTagsRDD = sc.textFile("/path/to/lastfm/user_taggedartists.dat")
.map(line => line.split("\t"))
.map(a => (a(0), a(2)))
.filter(_._1 != "userID")
val userTagsIDS = IndexedDatasetSpark.apply(userTagsRDD)(sc)

val userArtistsRDD = sc.textFile("/path/to/lastfm/user_artists.dat")
													.map(line => line.split("\t"))
													.map(a => (a(0), a(1)))
										      .filter(_._1 != "userID")
										      
val userArtistsIDS = IndexedDatasetSpark.apply(userArtistsRDD)(sc)

val userFriendsRDD = sc.textFile("/path/to/lastfm/user_friends.dat")
                          .map(line => line.split("\t"))
                          .map(a => (a(0), a(1)))
                          .filter(_._1 != "userID")
                          
val userFriendsIDS = IndexedDatasetSpark.apply(userFriendsRDD)(sc)
~~~
{: .language-scala}

How much easier was that?! In each RDD creations we:

Load our data using sc.textFile
    
    sc.textFile("/path/to/lastfm/user_taggedartists.dat")

Split the data into an array based on tabs (\t)

    .map(line => line.split("\t"))

Pull the userID column into the first position of the tuple, and the other attribute we want into the second position.

    .map(a => (a(0), a(1)))

Remove the header (the only line that will have “userID” in that position)

    .filter(_._1 != "userID")

Then we easily create an IndexedDataSet using the `apply` method. 
val userTagsIDS = IndexedDatasetSpark.apply(userTagsRDD)(sc)
Note the `(sc)` at the end. You may or may not need that.  `sc` is the SparkContext and should be passed as an implicit parameter, however the REPL environment (e.g. Mahout Shell or notebooks) has a hard time with the implicits, so I had to pass it explicitly.  

Now we compute our co-occurrence matrices:
```scala
import org.apache.mahout.math.cf.SimilarityAnalysis

val artistReccosLlrDrmListByArtist = SimilarityAnalysis.cooccurrencesIDSs(
			Array(userArtistsIDS, userTagsIDS, userFriendsIDS), 
						maxInterestingItemsPerThing = 20,
						maxNumInteractions = 500, 
						randomSeed = 1234)
```


Let’s see an example of how this would work-

First we have a small problem. If you look at our original input files, the userIDs, artistIDs, and tags were all integers. We loaded them as strings and if you look at the BiDictionaries associated with each IDS, you’ll see they map the original integers as strings to the integer indices of our matrix. Not super helpful.  There are other files which contain mappings from LastFM ID to human readable band and tag names.  I could have sorted this out in the begining but I chose to do it on the backside because it is a bit of clever Spark/Scala only needed to work around a quirk in this particular dataset.  We have to reverse map a few things if we want to input ‘human readable’ attributes, which I did.  If this doesn’t make sense, please don’t be discouraged- the important part was above, this is just some magic for working with this dataset in a pretty way. 

First I load, and create incore maps from the mapping files:

```scala
val artistMap = sc.textFile("/path/to/lastfm/artists.dat")
  .map(line => line.split("\t"))
  .map(a => (a(1), a(0)))
  .filter(_._1 != "name")
  .collect
  .toMap

val tagsMap = sc.textFile("/path/tolastfm/tags.dat")
  .map(line => line.split("\t"))
  .map(a => (a(1), a(0)))
  .filter(_._1 != "tagValue")
  .collect
  .toMap

```

This will create some `Map`s that I can use to type readable names for the artist and tags to create my ‘history’.

```scala
val kilroyUserArtists = svec( (userArtistsIDS.columnIDs.get(artistMap("Beck")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("David Bowie")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Gary Numan")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Less Than Jake")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Lou Reed")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Parliament")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Radiohead")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Seu Jorge")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("The Skatalites")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Reverend Horton Heat")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Talking Heads")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Tom Waits")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Waylon Jennings")).get, 1) ::
 (userArtistsIDS.columnIDs.get(artistMap("Wu-Tang Clan")).get, 1) :: Nil, 
 cardinality = userArtistsIDS.columnIDs.size
)



val kilroyUserTags = svec(
 (userTagsIDS.columnIDs.get(tagsMap("classical")).get, 1) ::
 (userTagsIDS.columnIDs.get(tagsMap("skacore")).get, 1) ::
 (userTagsIDS.columnIDs.get(tagsMap("why on earth is this just a bonus track")).get, 1) ::
 (userTagsIDS.columnIDs.get(tagsMap("punk rock")).get, 1) :: Nil,
 cardinality = userTagsIDS.columnIDs.size)
```

So what we have then is me typing in a name to `artistMap` where the keys are human readable names of my favorite bands, which returns the value which is the LastFM ID, which in turn is the key in the BiDictionary map, and returns the matrix position.  I’m making a sparse vector where I want the index at the value I just fetched (which in an awry way refers to the artist I specified) to have the value 1.  

Same idea for the tags. 

I now have two history vectors.  I didn’t make one for the users table, because I don’t have any friends on LastFM yet. That’s about to change though, because I’m about to have some friends recommended to me. 

val kilroysRecs = (artistReccosLlrDrmListByArtist(0).matrix %*% kilroyUserArtists + artistReccosLlrDrmListByArtist(1).matrix %*% kilroyUserTags).collect
Finally let’s sort that vector out and get some user ids and strengths. 
```scala
import org.apache.mahout.math.scalabindings.MahoutCollections._
import collection._
import JavaConversions._

// Which Users I should Be Friends with.
println(kilroysRecs(::, 0).toMap.toList.sortWith(_._2 > _._2).take(5))

```

`kilroysRecs` is actually a one column matrix, so we take that, and the convert it into something we can sort. We then take the top 5 suggestions.  Keep in mind, this will return the Mahout user ID, which you would also have to reverse map back to the lastFM userID.  The lastFM userID is just another Integer, and not particularly exciting so I left that out. 

If you wanted to recommend artists like a normal recommendation engine- you would change the first position in all of the input matrices to be “artistID”. This is left as an exercise to the user. 

[Full Scala Worksheet](cco-lastfm.scala)
