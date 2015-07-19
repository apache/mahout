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

package org.apache.mahout.math.cf

import org.apache.mahout.math._
import org.apache.mahout.math.indexeddataset.IndexedDataset
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import scala.collection.JavaConversions._
import org.apache.mahout.math.stats.LogLikelihood
import collection._
import org.apache.mahout.math.function.{VectorFunction, Functions}

import scala.util.Random


/**
 * Based on "Ted Dunnning & Ellen Friedman: Practical Machine Learning, Innovations in Recommendation",
 * available at http://www.mapr.com/practical-machine-learning
 *
 * see also "Sebastian Schelter, Christoph Boden, Volker Markl:
 * Scalable Similarity-Based Neighborhood Methods with MapReduce
 * ACM Conference on Recommender Systems 2012"
 */
object SimilarityAnalysis extends Serializable {

  /** Compares (Int,Double) pairs by the second value */
  private val orderByScore = Ordering.fromLessThan[(Int, Double)] { case ((_, score1), (_, score2)) => score1 > score2}

  /**
   * Calculates item (column-wise) similarity using the log-likelihood ratio on A'A, A'B, A'C, ...
   * and returns a list of similarity and cross-similarity matrices
   * @param drmARaw Primary interaction matrix
   * @param randomSeed when kept to a constant will make repeatable downsampling
   * @param maxInterestingItemsPerThing number of similar items to return per item, default: 50
   * @param maxNumInteractions max number of interactions after downsampling, default: 500
   * @return a list of [[org.apache.mahout.math.drm.DrmLike]] containing downsampled DRMs for cooccurrence and
   *         cross-cooccurrence
   */
  def cooccurrences(drmARaw: DrmLike[Int], randomSeed: Int = 0xdeadbeef, maxInterestingItemsPerThing: Int = 50,
                    maxNumInteractions: Int = 500, drmBs: Array[DrmLike[Int]] = Array()): List[DrmLike[Int]] = {

    implicit val distributedContext = drmARaw.context

    // backend allowed to optimize partitioning
    drmARaw.par(auto = true)

    // Apply selective downsampling, pin resulting matrix
    val drmA = sampleDownAndBinarize(drmARaw, randomSeed, maxNumInteractions)

    // num users, which equals the maximum number of interactions per item
    val numUsers = drmA.nrow.toInt

    // Compute & broadcast the number of interactions per thing in A
    val bcastInteractionsPerItemA = drmBroadcast(drmA.numNonZeroElementsPerColumn)

    // Compute cooccurrence matrix A'A
    val drmAtA = drmA.t %*% drmA

    // Compute loglikelihood scores and sparsify the resulting matrix to get the similarity matrix
    val drmSimilarityAtA = computeSimilarities(drmAtA, numUsers, maxInterestingItemsPerThing,
      bcastInteractionsPerItemA, bcastInteractionsPerItemA, crossCooccurrence = false)

    var similarityMatrices = List(drmSimilarityAtA)

    // Now look at cross cooccurrences
    for (drmBRaw <- drmBs) {
      // backend allowed to optimize partitioning
      drmBRaw.par(auto = true)

      // Down-sample and pin other interaction matrix
      val drmB = sampleDownAndBinarize(drmBRaw, randomSeed, maxNumInteractions).checkpoint()

      // Compute & broadcast the number of interactions per thing in B
      val bcastInteractionsPerThingB = drmBroadcast(drmB.numNonZeroElementsPerColumn)

      // Compute cross-cooccurrence matrix A'B
      val drmAtB = drmA.t %*% drmB

      val drmSimilarityAtB = computeSimilarities(drmAtB, numUsers, maxInterestingItemsPerThing,
        bcastInteractionsPerItemA, bcastInteractionsPerThingB)

      similarityMatrices = similarityMatrices :+ drmSimilarityAtB

      drmB.uncache()

      //debug
      val atbRows = drmSimilarityAtB.nrow
      val atbCols = drmSimilarityAtB.ncol
      val i = 0
    }

    // Unpin downsampled interaction matrix
    drmA.uncache()

    //debug
    val ataRows = drmSimilarityAtA.nrow
    val ataCols = drmSimilarityAtA.ncol
    val i = 0

    // Return list of similarity matrices
    similarityMatrices
  }

  /**
   * Calculates item (column-wise) similarity using the log-likelihood ratio on A'A, A'B, A'C, ... and returns
   * a list of similarity and cross-similarity matrices. Somewhat easier to use method, which handles the ID
   * dictionaries correctly
   * @param indexedDatasets first in array is primary/A matrix all others are treated as secondary
   * @param randomSeed use default to make repeatable, otherwise pass in system time or some randomizing seed
   * @param maxInterestingItemsPerThing max similarities per items
   * @param maxNumInteractions max number of input items per item
   * @return a list of [[org.apache.mahout.math.indexeddataset.IndexedDataset]] containing downsampled
   *         IndexedDatasets for cooccurrence and cross-cooccurrence
   */
  def cooccurrencesIDSs(indexedDatasets: Array[IndexedDataset],
      randomSeed: Int = 0xdeadbeef,
      maxInterestingItemsPerThing: Int = 50,
      maxNumInteractions: Int = 500):
    List[IndexedDataset] = {
    val drms = indexedDatasets.map(_.matrix.asInstanceOf[DrmLike[Int]])
    val primaryDrm = drms(0)
    val secondaryDrms = drms.drop(1)
    val coocMatrices = cooccurrences(primaryDrm, randomSeed, maxInterestingItemsPerThing,
      maxNumInteractions, secondaryDrms)
    val retIDSs = coocMatrices.iterator.zipWithIndex.map {
      case( drm, i ) =>
        indexedDatasets(0).create(drm, indexedDatasets(0).columnIDs, indexedDatasets(i).columnIDs)
    }
    retIDSs.toList
  }

  /**
   * Calculates row-wise similarity using the log-likelihood ratio on AA' and returns a DRM of rows and similar rows
   * @param drmARaw Primary interaction matrix
   * @param randomSeed when kept to a constant will make repeatable downsampling
   * @param maxInterestingSimilaritiesPerRow number of similar items to return per item, default: 50
   * @param maxNumInteractions max number of interactions after downsampling, default: 500
   */
  def rowSimilarity(drmARaw: DrmLike[Int], randomSeed: Int = 0xdeadbeef, maxInterestingSimilaritiesPerRow: Int = 50,
                    maxNumInteractions: Int = 500): DrmLike[Int] = {

    implicit val distributedContext = drmARaw.context

    // backend allowed to optimize partitioning
    drmARaw.par(auto = true)

    // Apply selective downsampling, pin resulting matrix
    val drmA = sampleDownAndBinarize(drmARaw, randomSeed, maxNumInteractions)

    // num columns, which equals the maximum number of interactions per item
    val numCols = drmA.ncol

    // Compute & broadcast the number of interactions per row in A
    val bcastInteractionsPerItemA = drmBroadcast(drmA.numNonZeroElementsPerRow)

    // Compute row similarity cooccurrence matrix AA'
    val drmAAt = drmA %*% drmA.t

    // Compute loglikelihood scores and sparsify the resulting matrix to get the similarity matrix
    val drmSimilaritiesAAt = computeSimilarities(drmAAt, numCols, maxInterestingSimilaritiesPerRow,
      bcastInteractionsPerItemA, bcastInteractionsPerItemA, crossCooccurrence = false)

    drmSimilaritiesAAt
  }

  /**
   * Calculates row-wise similarity using the log-likelihood ratio on AA' and returns a drm of rows and similar rows.
   * Uses IndexedDatasets, which handle external ID dictionaries properly
   * @param indexedDataset compare each row to every other
   * @param randomSeed  use default to make repeatable, otherwise pass in system time or some randomizing seed
   * @param maxInterestingSimilaritiesPerRow max elements returned in each row
   * @param maxObservationsPerRow max number of input elements to use
   */
  def rowSimilarityIDS(indexedDataset: IndexedDataset, randomSeed: Int = 0xdeadbeef,
      maxInterestingSimilaritiesPerRow: Int = 50,
      maxObservationsPerRow: Int = 500):
    IndexedDataset = {
    val coocMatrix = rowSimilarity(indexedDataset.matrix, randomSeed, maxInterestingSimilaritiesPerRow,
      maxObservationsPerRow)
    indexedDataset.create(coocMatrix, indexedDataset.rowIDs, indexedDataset.rowIDs)
  }

   /** Compute loglikelihood ratio see http://tdunning.blogspot.de/2008/03/surprise-and-coincidence.html for details */
  def logLikelihoodRatio(numInteractionsWithA: Long, numInteractionsWithB: Long,
    numInteractionsWithAandB: Long, numInteractions: Long) = {

    val k11 = numInteractionsWithAandB
    val k12 = numInteractionsWithA - numInteractionsWithAandB
    val k21 = numInteractionsWithB - numInteractionsWithAandB
    val k22 = numInteractions - numInteractionsWithA - numInteractionsWithB + numInteractionsWithAandB

    LogLikelihood.logLikelihoodRatio(k11, k12, k21, k22)

  }

  def computeSimilarities(drm: DrmLike[Int], numUsers: Int, maxInterestingItemsPerThing: Int,
                        bcastNumInteractionsB: BCast[Vector], bcastNumInteractionsA: BCast[Vector],
                        crossCooccurrence: Boolean = true) = {
    drm.mapBlock() {
      case (keys, block) =>

        val llrBlock = block.like()
        val numInteractionsB: Vector = bcastNumInteractionsB
        val numInteractionsA: Vector = bcastNumInteractionsA

        for (index <- 0 until keys.size) {

          val thingB = keys(index)

          // PriorityQueue to select the top-k items
          val topItemsPerThing = new mutable.PriorityQueue[(Int, Double)]()(orderByScore)

          block(index, ::).nonZeroes().foreach { elem =>
            val thingA = elem.index
            val cooccurrences = elem.get

            // exclude co-occurrences of the item with itself
            if (crossCooccurrence || thingB != thingA) {
              // Compute loglikelihood ratio
              val llr = logLikelihoodRatio(numInteractionsB(thingB).toLong, numInteractionsA(thingA).toLong,
                cooccurrences.toLong, numUsers)

              val candidate = thingA -> llr

              // legacy hadoop code maps values to range (0..1) via
              // val normailizedLLR = 1.0 - (1.0 / (1.0 + llr))
              // val candidate = thingA -> normailizedLLR

              // Enqueue item with score, if belonging to the top-k
              if (topItemsPerThing.size < maxInterestingItemsPerThing) {
                topItemsPerThing.enqueue(candidate)
              } else if (orderByScore.lt(candidate, topItemsPerThing.head)) {
                topItemsPerThing.dequeue()
                topItemsPerThing.enqueue(candidate)
              }
            }
          }

          // Add top-k interesting items to the output matrix
          topItemsPerThing.dequeueAll.foreach {
            case (otherThing, llrScore) =>
              llrBlock(index, otherThing) = llrScore
          }
        }

        keys -> llrBlock
    }
  }

  /**
   * Selectively downsample rows and items with an anomalous amount of interactions, inspired by
   * https://github.com/tdunning/in-memory-cooccurrence/blob/master/src/main/java/com/tdunning/cooc/Analyze.java
   *
   * additionally binarizes input matrix, as we're only interesting in knowing whether interactions happened or not
   * @param drmM matrix to downsample
   * @param seed random number generator seed, keep to a constant if repeatability is neccessary
   * @param maxNumInteractions number of elements in a row of the returned matrix
   * @return the downsampled DRM
   */
  def sampleDownAndBinarize(drmM: DrmLike[Int], seed: Int, maxNumInteractions: Int) = {

    implicit val distributedContext = drmM.context

    // Pin raw interaction matrix
    val drmI = drmM.checkpoint()

    // Broadcast vector containing the number of interactions with each thing
    val bcastNumInteractions = drmBroadcast(drmI.numNonZeroElementsPerColumn)

    val downSampledDrmI = drmI.mapBlock() {
      case (keys, block) =>
        val numInteractions: Vector = bcastNumInteractions

        // Use a hash of the unique first key to seed the RNG, makes this computation repeatable in case of
        //failures
        val random = new Random(MurmurHash.hash(keys(0), seed))

        val downsampledBlock = block.like()

        // Downsample the interaction vector of each row
        for (rowIndex <- 0 until keys.size) {

          val interactionsInRow = block(rowIndex, ::)

          val numInteractionsPerRow = interactionsInRow.getNumNonZeroElements()

          val perRowSampleRate = math.min(maxNumInteractions, numInteractionsPerRow) / numInteractionsPerRow

          interactionsInRow.nonZeroes().foreach { elem =>
            val numInteractionsWithThing = numInteractions(elem.index)
            val perThingSampleRate = math.min(maxNumInteractions, numInteractionsWithThing) / numInteractionsWithThing

            if (random.nextDouble() <= math.min(perRowSampleRate, perThingSampleRate)) {
              // We ignore the original interaction value and create a binary 0-1 matrix
              // as we only consider whether interactions happened or did not happen
              downsampledBlock(rowIndex, elem.index) = 1
            }
          }
        }

        keys -> downsampledBlock
    }

    // Unpin raw interaction matrix
    drmI.uncache()

    downSampledDrmI
  }
}
