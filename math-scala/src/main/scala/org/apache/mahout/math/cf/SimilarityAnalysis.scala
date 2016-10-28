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

  lazy val defaultParOpts = ParOpts()

  /**
   * Calculates item (column-wise) similarity using the log-likelihood ratio on A'A, A'B, A'C, ...
   * and returns a list of similarity and cross-similarity matrices
    *
    * @param drmARaw Primary interaction matrix
   * @param randomSeed when kept to a constant will make repeatable downsampling
   * @param maxInterestingItemsPerThing number of similar items to return per item, default: 50
   * @param maxNumInteractions max number of interactions after downsampling, default: 500
   * @param parOpts partitioning params for drm.par(...)
   * @return a list of [[org.apache.mahout.math.drm.DrmLike]] containing downsampled DRMs for cooccurrence and
   *         cross-cooccurrence
   */
  def cooccurrences(
    drmARaw: DrmLike[Int],
    randomSeed: Int = 0xdeadbeef,
    maxInterestingItemsPerThing: Int = 50,
    maxNumInteractions: Int = 500,
    drmBs: Array[DrmLike[Int]] = Array(),
    parOpts: ParOpts = defaultParOpts)
    : List[DrmLike[Int]] = {

    implicit val distributedContext = drmARaw.context

    // backend partitioning defaults to 'auto', which is often better decided by calling funciton
    // todo:  this should ideally be different per drm
    drmARaw.par( min = parOpts.minPar, exact = parOpts.exactPar, auto = parOpts.autoPar)

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
      // backend partitioning defaults to 'auto', which is often better decided by calling funciton
      // todo:  this should ideally be different per drm
      drmARaw.par( min = parOpts.minPar, exact = parOpts.exactPar, auto = parOpts.autoPar)

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
    }

    // Unpin downsampled interaction matrix
    drmA.uncache()

    // Return list of similarity matrices
    similarityMatrices
  }

  /**
   * Calculates item (column-wise) similarity using the log-likelihood ratio on A'A, A'B, A'C, ... and returns
   * a list of similarity and cross-similarity matrices. Somewhat easier to use method, which handles the ID
   * dictionaries correctly
   *
   * @param indexedDatasets first in array is primary/A matrix all others are treated as secondary
   * @param randomSeed use default to make repeatable, otherwise pass in system time or some randomizing seed
   * @param maxInterestingItemsPerThing max similarities per items
   * @param maxNumInteractions max number of input items per item
   * @param parOpts partitioning params for drm.par(...)
   * @return a list of [[org.apache.mahout.math.indexeddataset.IndexedDataset]] containing downsampled
   *         IndexedDatasets for cooccurrence and cross-cooccurrence
   */
  def cooccurrencesIDSs(
    indexedDatasets: Array[IndexedDataset],
    randomSeed: Int = 0xdeadbeef,
    maxInterestingItemsPerThing: Int = 50,
    maxNumInteractions: Int = 500,
    parOpts: ParOpts = defaultParOpts):
    List[IndexedDataset] = {
    val drms = indexedDatasets.map(_.matrix.asInstanceOf[DrmLike[Int]])
    val primaryDrm = drms(0)
    val secondaryDrms = drms.drop(1)
    val coocMatrices = cooccurrences(primaryDrm, randomSeed, maxInterestingItemsPerThing,
      maxNumInteractions, secondaryDrms, parOpts)
    val retIDSs = coocMatrices.iterator.zipWithIndex.map {
      case( drm, i ) =>
        indexedDatasets(0).create(drm, indexedDatasets(0).columnIDs, indexedDatasets(i).columnIDs)
    }
    retIDSs.toList
  }

  /**
    * Calculates item (column-wise) similarity using the log-likelihood ratio on A'A, A'B, A'C, ... and returns
    * a list of similarity and cross-occurrence matrices. Somewhat easier to use method, which handles the ID
    * dictionaries correctly and contains info about downsampling in each model calc.
    *
    * @param datasets first in array is primary/A matrix all others are treated as secondary, includes information
    *                 used to downsample the input drm as well as the output llr(A'A), llr(A'B). The information
    *                 is contained in each dataset in the array and applies to the model calculation of A' with
    *                 the dataset. Todo: ignoring absolute threshold for now.
    * @param randomSeed use default to make repeatable, otherwise pass in system time or some randomizing seed
    * @param parOpts partitioning params for drm.par(...)
    * @return a list of [[org.apache.mahout.math.indexeddataset.IndexedDataset]] containing downsampled
    *         IndexedDatasets for cooccurrence and cross-cooccurrence
    */
  def crossOccurrenceDownsampled(
    datasets: List[DownsamplableCrossOccurrenceDataset],
    randomSeed: Int = 0xdeadbeef):
    List[IndexedDataset] = {


    val crossDatasets = datasets.drop(1) // drop A
    val primaryDataset = datasets.head // use A throughout
    val drmARaw = primaryDataset.iD.matrix

    implicit val distributedContext = primaryDataset.iD.matrix.context

    // backend partitioning defaults to 'auto', which is often better decided by calling funciton
    val parOptsA = primaryDataset.parOpts.getOrElse(defaultParOpts)
    drmARaw.par( min = parOptsA.minPar, exact = parOptsA.exactPar, auto = parOptsA.autoPar)

    // Apply selective downsampling, pin resulting matrix
    val drmA = sampleDownAndBinarize(drmARaw, randomSeed, primaryDataset.maxElementsPerRow)

    // num users, which equals the maximum number of interactions per item
    val numUsers = drmA.nrow.toInt

    // Compute & broadcast the number of interactions per thing in A
    val bcastInteractionsPerItemA = drmBroadcast(drmA.numNonZeroElementsPerColumn)

    // Compute cooccurrence matrix A'A
    val drmAtA = drmA.t %*% drmA

    // Compute loglikelihood scores and sparsify the resulting matrix to get the similarity matrix
    val drmSimilarityAtA = computeSimilarities(drmAtA, numUsers, primaryDataset.maxInterestingElements,
      bcastInteractionsPerItemA, bcastInteractionsPerItemA, crossCooccurrence = false,
      minLLROpt = primaryDataset.minLLROpt)

    var similarityMatrices = List(drmSimilarityAtA)

    // Now look at cross cooccurrences
    for (dataset <- crossDatasets) {
      // backend partitioning defaults to 'auto', which is often better decided by calling funciton
      val parOptsB = dataset.parOpts.getOrElse(defaultParOpts)
      dataset.iD.matrix.par(min = parOptsB.minPar, exact = parOptsB.exactPar, auto = parOptsB.autoPar)

      // Downsample and pin other interaction matrix
      val drmB = sampleDownAndBinarize(dataset.iD.matrix, randomSeed, dataset.maxElementsPerRow).checkpoint()

      // Compute & broadcast the number of interactions per thing in B
      val bcastInteractionsPerThingB = drmBroadcast(drmB.numNonZeroElementsPerColumn)

      // Compute cross-cooccurrence matrix A'B
      val drmAtB = drmA.t %*% drmB

      val drmSimilarityAtB = computeSimilarities(drmAtB, numUsers, dataset.maxInterestingElements,
        bcastInteractionsPerItemA, bcastInteractionsPerThingB, minLLROpt = dataset.minLLROpt)

      similarityMatrices = similarityMatrices :+ drmSimilarityAtB

      drmB.uncache()
    }

    // Unpin downsampled interaction matrix
    drmA.uncache()

    // Return list of datasets
    val retIDSs = similarityMatrices.iterator.zipWithIndex.map {
      case( drm, i ) =>
        datasets(0).iD.create(drm, datasets(0).iD.columnIDs, datasets(i).iD.columnIDs)
    }
    retIDSs.toList

  }

  /**
   * Calculates row-wise similarity using the log-likelihood ratio on AA' and returns a DRM of rows and similar rows
   *
   * @param drmARaw Primary interaction matrix
   * @param randomSeed when kept to a constant will make repeatable downsampling
   * @param maxInterestingSimilaritiesPerRow number of similar items to return per item, default: 50
   * @param maxNumInteractions max number of interactions after downsampling, default: 500
   * @param parOpts partitioning options used for drm.par(...)
   */
  def rowSimilarity(
    drmARaw: DrmLike[Int],
    randomSeed: Int = 0xdeadbeef,
    maxInterestingSimilaritiesPerRow: Int = 50,
    maxNumInteractions: Int = 500,
    parOpts: ParOpts = defaultParOpts): DrmLike[Int] = {

    implicit val distributedContext = drmARaw.context

    // backend partitioning defaults to 'auto', which is often better decided by calling funciton
    // todo: should this ideally be different per drm?
    drmARaw.par(min = parOpts.minPar, exact = parOpts.exactPar, auto = parOpts.autoPar)

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
   *
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

  def computeSimilarities(
    drm: DrmLike[Int],
    numUsers: Int,
    maxInterestingItemsPerThing: Int,
    bcastNumInteractionsB: BCast[Vector],
    bcastNumInteractionsA: BCast[Vector],
    crossCooccurrence: Boolean = true,
    minLLROpt: Option[Double] = None) = {

    //val minLLR = minLLROpt.getOrElse(0.0d) // accept all values if not specified

    val minLLR = minLLROpt

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
              if(minLLR.isEmpty || llr >= minLLR.get) { // llr threshold takes precedence over max per row
                if (topItemsPerThing.size < maxInterestingItemsPerThing) {
                  topItemsPerThing.enqueue(candidate)
                } else if (orderByScore.lt(candidate, topItemsPerThing.head)) {
                  topItemsPerThing.dequeue()
                  topItemsPerThing.enqueue(candidate)
                }
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
   *
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

case class ParOpts( // this will contain the default `par` params except for auto = true
  minPar: Int = -1,
  exactPar: Int = -1,
  autoPar: Boolean = true)

/* Used to pass in data and params for downsampling the input data as well as output A'A, A'B, etc. */
case class DownsamplableCrossOccurrenceDataset(
  iD: IndexedDataset,
  maxElementsPerRow: Int = 500, // usually items per user in the input dataset, used to ramdomly downsample
  maxInterestingElements: Int = 50, // number of items/columns to keep in the A'A, A'B etc. where iD == A, B, C ...
  minLLROpt: Option[Double] = None, // absolute threshold, takes precedence over maxInterestingElements if present
  parOpts: Option[ParOpts] = None) // these can be set per dataset and are applied to each of the drms
                                // in crossOccurrenceDownsampled

