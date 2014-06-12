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

package org.apache.mahout.cf

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import scala.collection.JavaConversions._
import org.apache.mahout.math.stats.LogLikelihood
import collection._
import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.function.{VectorFunction, Functions}


/**
 * based on "Ted Dunnning & Ellen Friedman: Practical Machine Learning, Innovations in Recommendation",
 * available at http://www.mapr.com/practical-machine-learning
 *
 * see also "Sebastian Schelter, Christoph Boden, Volker Markl:
 * Scalable Similarity-Based Neighborhood Methods with MapReduce
 * ACM Conference on Recommender Systems 2012"
 */
object CooccurrenceAnalysis extends Serializable {

  /** Compares (Int,Double) pairs by the second value */
  private val orderByScore = Ordering.fromLessThan[(Int, Double)] { case ((_, score1), (_, score2)) => score1 > score2}

  def cooccurrences(drmARaw: DrmLike[Int], randomSeed: Int = 0xdeadbeef, maxInterestingItemsPerThing: Int = 50,
                    maxNumInteractions: Int = 500, drmBs: Array[DrmLike[Int]] = Array()): List[DrmLike[Int]] = {

    implicit val distributedContext = drmARaw.context

    // Apply selective downsampling, pin resulting matrix
    val drmA = sampleDownAndBinarize(drmARaw, randomSeed, maxNumInteractions)

    // num users, which equals the maximum number of interactions per item
    val numUsers = drmA.nrow.toInt

    // Compute & broadcast the number of interactions per thing in A
    val bcastInteractionsPerItemA = drmBroadcast(drmA.numNonZeroElementsPerColumn)

    // Compute co-occurrence matrix A'A
    val drmAtA = drmA.t %*% drmA

    // Compute loglikelihood scores and sparsify the resulting matrix to get the indicator matrix
    val drmIndicatorsAtA = computeIndicators(drmAtA, numUsers, maxInterestingItemsPerThing, bcastInteractionsPerItemA,
      bcastInteractionsPerItemA, crossCooccurrence = false)

    var indicatorMatrices = List(drmIndicatorsAtA)

    // Now look at cross-co-occurrences
    for (drmBRaw <- drmBs) {
      // Down-sample and pin other interaction matrix
      val drmB = sampleDownAndBinarize(drmBRaw, randomSeed, maxNumInteractions).checkpoint()

      // Compute & broadcast the number of interactions per thing in B
      val bcastInteractionsPerThingB = drmBroadcast(drmB.numNonZeroElementsPerColumn)

      // Compute cross-co-occurrence matrix B'A
      val drmBtA = drmB.t %*% drmA

      val drmIndicatorsBtA = computeIndicators(drmBtA, numUsers, maxInterestingItemsPerThing,
        bcastInteractionsPerThingB, bcastInteractionsPerItemA)

      indicatorMatrices = indicatorMatrices :+ drmIndicatorsBtA

      drmB.uncache()
    }

    // Unpin downsampled interaction matrix
    drmA.uncache()

    // Return list of indicator matrices
    indicatorMatrices
  }

  /**
   * Compute loglikelihood ratio
   * see http://tdunning.blogspot.de/2008/03/surprise-and-coincidence.html for details
   **/
  def loglikelihoodRatio(numInteractionsWithA: Long, numInteractionsWithB: Long,
                         numInteractionsWithAandB: Long, numInteractions: Long) = {

    val k11 = numInteractionsWithAandB
    val k12 = numInteractionsWithA - numInteractionsWithAandB
    val k21 = numInteractionsWithB - numInteractionsWithAandB
    val k22 = numInteractions - numInteractionsWithA - numInteractionsWithB + numInteractionsWithAandB

    LogLikelihood.logLikelihoodRatio(k11, k12, k21, k22)
  }

  def computeIndicators(drmBtA: DrmLike[Int], numUsers: Int, maxInterestingItemsPerThing: Int,
                        bcastNumInteractionsB: BCast[Vector], bcastNumInteractionsA: BCast[Vector],
                        crossCooccurrence: Boolean = true) = {
    drmBtA.mapBlock() {
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
              val llrRatio = loglikelihoodRatio(numInteractionsB(thingB).toLong, numInteractionsA(thingA).toLong,
                cooccurrences.toLong, numUsers)
              val candidate = thingA -> llrRatio

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
   * Selectively downsample users and things with an anomalous amount of interactions, inspired by
   * https://github.com/tdunning/in-memory-cooccurrence/blob/master/src/main/java/com/tdunning/cooc/Analyze.java
   *
   * additionally binarizes input matrix, as we're only interesting in knowing whether interactions happened or not
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

        // Use a hash of the unique first key to seed the RNG, makes this computation repeatable in case of failures
        val random = RandomUtils.getRandom(MurmurHash.hash(keys(0), seed))

        val downsampledBlock = block.like()

        // Downsample the interaction vector of each user
        for (userIndex <- 0 until keys.size) {

          val interactionsOfUser = block(userIndex, ::)

          val numInteractionsOfUser = interactionsOfUser.getNumNonZeroElements()

          val perUserSampleRate = math.min(maxNumInteractions, numInteractionsOfUser) / numInteractionsOfUser

          interactionsOfUser.nonZeroes().foreach { elem =>
            val numInteractionsWithThing = numInteractions(elem.index)
            val perThingSampleRate = math.min(maxNumInteractions, numInteractionsWithThing) / numInteractionsWithThing

            if (random.nextDouble() <= math.min(perUserSampleRate, perThingSampleRate)) {
              // We ignore the original interaction value and create a binary 0-1 matrix
              // as we only consider whether interactions happened or did not happen
              downsampledBlock(userIndex, elem.index) = 1
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
