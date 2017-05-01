
/**
  * Created by rawkintrevo on 4/5/17.
  */

// Only need these to intelliJ doesn't whine

import org.apache.mahout.drivers.ItemSimilarityDriver.parser
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
val conf = new SparkConf().setAppName("Simple Application")
val sc = new SparkContext(conf)

implicit val sdc: org.apache.mahout.sparkbindings.SparkDistributedContext = sc2sdc(sc)


// </pandering to intellij>

// http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
// start mahout shell like this: $MAHOUT_HOME/bin/mahout spark-shell

import org.apache.mahout.sparkbindings.indexeddataset.IndexedDatasetSpark

// We need to turn our raw text files into RDD[(String, String)] 
val userTagsRDD = sc.textFile("/home/rawkintrevo/gits/MahoutExamples/data/lastfm/user_taggedartists.dat").map(line => line.split("\t")).map(a => (a(0), a(2))).filter(_._1 != "userID")
val userTagsIDS = IndexedDatasetSpark.apply(userTagsRDD)(sc)

val userArtistsRDD = sc.textFile("/home/rawkintrevo/gits/MahoutExamples/data/lastfm/user_artists.dat").map(line => line.split("\t")).map(a => (a(0), a(1))).filter(_._1 != "userID")
val userArtistsIDS = IndexedDatasetSpark.apply(userArtistsRDD)(sc)

val userFriendsRDD = sc.textFile("/home/rawkintrevo/gits/MahoutExamples/data/lastfm/user_friends.dat").map(line => line.split("\t")).map(a => (a(0), a(1))).filter(_._1 != "userID")
val userFriendsIDS = IndexedDatasetSpark.apply(userFriendsRDD)(sc)

import org.apache.mahout.math.cf.SimilarityAnalysis

val artistReccosLlrDrmListByArtist = SimilarityAnalysis.cooccurrencesIDSs(Array(userArtistsIDS, userTagsIDS, userFriendsIDS), maxInterestingItemsPerThing = 20, maxNumInteractions = 500, randomSeed = 1234)

// Anonymous User

val artistMap = sc.textFile("/home/rawkintrevo/gits/MahoutExamples/data/lastfm/artists.dat").map(line => line.split("\t")).map(a => (a(1), a(0))).filter(_._1 != "name").collect.toMap
val tagsMap = sc.textFile("/home/rawkintrevo/gits/MahoutExamples/data/lastfm/tags.dat").map(line => line.split("\t")).map(a => (a(1), a(0))).filter(_._1 != "tagValue").collect.toMap

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