/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

/*
 * Download data from: http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
 * then run this in the mahout shell.
 */

import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark

// We need to turn our raw text files into RDD[(String, String)] 
val userTagsRDD = sc.textFile("/path/to/lastfm/user_taggedartists.dat").map(line => line.split("\t")).map(a => (a(0), a(2))).filter(_._1 != "userID")
val userTagsIDS = IndexedDatasetSpark.apply(userTagsRDD)(sc)

val userArtistsRDD = sc.textFile("/path/to/lastfm/user_artists.dat").map(line => line.split("\t")).map(a => (a(0), a(1))).filter(_._1 != "userID")
val userArtistsIDS = IndexedDatasetSpark.apply(userArtistsRDD)(sc)

val userFriendsRDD = sc.textFile("/path/to/data/lastfm/user_friends.dat").map(line => line.split("\t")).map(a => (a(0), a(1))).filter(_._1 != "userID")
val userFriendsIDS = IndexedDatasetSpark.apply(userFriendsRDD)(sc)

val primaryIDS = userFriendsIDS
val secondaryActionRDDs = List(userArtistsRDD, userTagsRDD)

import org.apache.mahout.math.indexeddataset.{IndexedDataset, BiDictionary}

def adjustRowCardinality(rowCardinality: Integer, datasetA: IndexedDataset): IndexedDataset = {
  val returnedA = if (rowCardinality != datasetA.matrix.nrow) datasetA.newRowCardinality(rowCardinality)
  else datasetA // this guarantees matching cardinality

  returnedA
}

var rowCardinality = primaryIDS.rowIDs.size

val secondaryActionIDS: Array[IndexedDataset] = new Array[IndexedDataset](secondaryActionRDDs.length)
for (i <- secondaryActionRDDs.indices) {

  val bcPrimaryRowIDs = sc.broadcast(primaryIDS.rowIDs)
  bcPrimaryRowIDs.value

  val tempRDD = secondaryActionRDDs(i).filter(a => bcPrimaryRowIDs.value.contains(a._1))

  var tempIDS = IndexedDatasetSpark.apply(tempRDD, existingRowIDs = Some(primaryIDS.rowIDs))(sc)
  secondaryActionIDS(i) = adjustRowCardinality(rowCardinality,tempIDS)
}

import org.apache.mahout.math.cf.SimilarityAnalysis

val artistReccosLlrDrmListByArtist = SimilarityAnalysis.cooccurrencesIDSs(
  Array(primaryIDS, secondaryActionIDS(0), secondaryActionIDS(1)),
  maxInterestingItemsPerThing = 20,
  maxNumInteractions = 500,
  randomSeed = 1234)
// Anonymous User

val artistMap = sc.textFile("/path/to/lastfm/artists.dat").map(line => line.split("\t")).map(a => (a(1), a(0))).filter(_._1 != "name").collect.toMap
val tagsMap = sc.textFile("/path/to/lastfm/tags.dat").map(line => line.split("\t")).map(a => (a(1), a(0))).filter(_._1 != "tagValue").collect.toMap

// Watch your skin- you're not wearing armour. (This will fail on misspelled artists
// This is neccessary because the ids are integer-strings already, and for this demo I didn't want to chance them to Integer types (bc more often you'll have strings).
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
  (userArtistsIDS.columnIDs.get(artistMap("Wu-Tang Clan")).get, 1) :: Nil, cardinality = userArtistsIDS.columnIDs.size
)

val kilroyUserTags = svec(
  (userTagsIDS.columnIDs.get(tagsMap("classical")).get, 1) ::
  (userTagsIDS.columnIDs.get(tagsMap("skacore")).get, 1) ::
  (userTagsIDS.columnIDs.get(tagsMap("why on earth is this just a bonus track")).get, 1) ::
  (userTagsIDS.columnIDs.get(tagsMap("punk rock")).get, 1) :: Nil, cardinality = userTagsIDS.columnIDs.size)

val kilroysRecs = (artistReccosLlrDrmListByArtist(0).matrix %*% kilroyUserArtists + artistReccosLlrDrmListByArtist(1).matrix %*% kilroyUserTags).collect


import org.apache.mahout.math.scalabindings.MahoutCollections._
import collection._
import JavaConversions._

// Which Users I should Be Friends with.
println(kilroysRecs(::, 0).toMap.toList.sortWith(_._2 > _._2).take(5))

/**
  * So there you have it- the basis for a new dating/friend finding app based on musical preferences which
  * is actually a pretty dope idea.
  *
  * Solving for which bands a user might like is left as an exercise to the reader.
  */