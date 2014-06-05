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

package org.apache.mahout.cf.examples

import scala.io.Source
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings._

import org.apache.mahout.cf.CooccurrenceAnalysis._
import scala.collection.JavaConversions._

/**
 * The Epinions dataset contains ratings from users to items and a trust-network between the users.
 * We use co-occurrence analysis to compute "users who like these items, also like that items" and
 * "users who trust these users, like that items"
 *
 * Download and unpack the dataset files from:
 *
 * http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2
 * http://www.trustlet.org/datasets/downloaded_epinions/trust_data.txt.bz2
 **/
object RunCrossCooccurrenceAnalysisOnEpinions {

  def main(args: Array[String]): Unit = {

    if (args.length == 0) {
      println("Usage: RunCooccurrenceAnalysisOnMovielens1M <path-to-dataset-folder>")
      println("Download the dataset from http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2 and")
      println("http://www.trustlet.org/datasets/downloaded_epinions/trust_data.txt.bz2")
      sys.exit(-1)
    }

    val datasetDir = args(0)

    val epinionsRatings = new SparseMatrix(49290, 139738)

    var firstLineSkipped = false
    for (line <- Source.fromFile(datasetDir + "/ratings_data.txt").getLines()) {
      if (line.contains(' ') && firstLineSkipped) {
        val tokens = line.split(' ')
        val userID = tokens(0).toInt - 1
        val itemID = tokens(1).toInt - 1
        val rating = tokens(2).toDouble
        epinionsRatings(userID, itemID) = rating
      }
      firstLineSkipped = true
    }

    val epinionsTrustNetwork = new SparseMatrix(49290, 49290)
    firstLineSkipped = false
    for (line <- Source.fromFile(datasetDir + "/trust_data.txt").getLines()) {
      if (line.contains(' ') && firstLineSkipped) {
        val tokens = line.trim.split(' ')
        val userID = tokens(0).toInt - 1
        val trustedUserId = tokens(1).toInt - 1
        epinionsTrustNetwork(userID, trustedUserId) = 1
      }
      firstLineSkipped = true
    }

    System.setProperty("spark.kryo.referenceTracking", "false")
    System.setProperty("spark.kryoserializer.buffer.mb", "100")
/* to run on local, can provide number of core by changing to local[4] */
    implicit val distributedContext = mahoutSparkContext(masterUrl = "local", appName = "MahoutLocalContext",
      customJars = Traversable.empty[String])

    /* to run on a Spark cluster provide the Spark Master URL
    implicit val distributedContext = mahoutSparkContext(masterUrl = "spark://occam4:7077", appName = "MahoutClusteredContext",
      customJars = Traversable.empty[String])
*/
    val drmEpinionsRatings = drmParallelize(epinionsRatings, numPartitions = 2)
    val drmEpinionsTrustNetwork = drmParallelize(epinionsTrustNetwork, numPartitions = 2)

    val indicatorMatrices = cooccurrences(drmEpinionsRatings, randomSeed = 0xdeadbeef,
        maxInterestingItemsPerThing = 100, maxNumInteractions = 500, Array(drmEpinionsTrustNetwork))

/* local storage */
    RecommendationExamplesHelper.saveIndicatorMatrix(indicatorMatrices(0),
        "/tmp/co-occurrence-on-epinions/indicators-item-item/")
    RecommendationExamplesHelper.saveIndicatorMatrix(indicatorMatrices(1),
        "/tmp/co-occurrence-on-epinions/indicators-trust-item/")

/*  To run on HDFS put your path to the data here, example of fully qualified path on my cluster provided
    RecommendationExamplesHelper.saveIndicatorMatrix(indicatorMatrices(0),
      "hdfs://occam4:54310/user/pat/xrsj/indicators-item-item/")
    RecommendationExamplesHelper.saveIndicatorMatrix(indicatorMatrices(1),
      "hdfs://occam4:54310/user/pat/xrsj/indicators-trust-item/")
*/
    distributedContext.close()

    println("Saved indicators to /tmp/co-occurrence-on-epinions/")
  }
}

/**
 * The movielens1M dataset contains movie ratings, we use co-occurrence analysis to compute
 * "users who like these movies, also like that movies"
 *
 * Download and unpack the dataset files from:
 * http://files.grouplens.org/datasets/movielens/ml-1m.zip
 */
object RunCooccurrenceAnalysisOnMovielens1M {

  def main(args: Array[String]): Unit = {

    if (args.length == 0) {
      println("Usage: RunCooccurrenceAnalysisOnMovielens1M <path-to-dataset-folder>")
      println("Download the dataset from http://files.grouplens.org/datasets/movielens/ml-1m.zip")
      sys.exit(-1)
    }

    val datasetDir = args(0)

    System.setProperty("spark.kryo.referenceTracking", "false")
    System.setProperty("spark.kryoserializer.buffer.mb", "100")

    implicit val sc = mahoutSparkContext(masterUrl = "local", appName = "MahoutLocalContext",
      customJars = Traversable.empty[String])

    System.setProperty("mahout.math.AtA.maxInMemNCol", 4000.toString)

    val movielens = new SparseMatrix(6040, 3952)

    for (line <- Source.fromFile(datasetDir + "/ratings.dat").getLines()) {
      val tokens = line.split("::")
      val userID = tokens(0).toInt - 1
      val itemID = tokens(1).toInt - 1
      val rating = tokens(2).toDouble
      movielens(userID, itemID) = rating
    }

    val drmMovielens = drmParallelize(movielens, numPartitions = 2)

    val indicatorMatrix = cooccurrences(drmMovielens).head

    RecommendationExamplesHelper.saveIndicatorMatrix(indicatorMatrix,
        "/tmp/co-occurrence-on-movielens/indicators-item-item/")

    sc.stop()

    println("Saved indicators to /tmp/co-occurrence-on-movielens/")
  }
}

object RecommendationExamplesHelper {

  def saveIndicatorMatrix(indicatorMatrix: DrmLike[Int], path: String) = {
    indicatorMatrix.rdd.flatMap({ case (thingID, itemVector) =>
        for (elem <- itemVector.nonZeroes()) yield { thingID + '\t' + elem.index }
      })
      .saveAsTextFile(path)
  }
}
