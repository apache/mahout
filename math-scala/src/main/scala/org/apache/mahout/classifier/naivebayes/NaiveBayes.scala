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

package org.apache.mahout.classifier.naivebayes


import org.apache.mahout.math._
import scalabindings._
import scalabindings.RLikeOps._
import drm.RLikeDrmOps._
import drm._
import scala.reflect.ClassTag
import scala.language.asInstanceOf
import collection._
import JavaConversions._


import org.apache.mahout.classifier.naivebayes.training.ComplementaryThetaTrainer

/**
 * Distributed training of a Naive Bayes model. Follows the approach presented in Rennie et.al.: Tackling the poor
 * assumptions of Naive Bayes Text classifiers, ICML 2003, http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
 */
object NaiveBayes {

  /** default value for the Laplacian smoothing parameter */
  def defaultAlphaI = 1.0f

  // function to extract categories from string keys
  type CategoryParser = String => String

  /** Default: seq2Sparse Categories are Stored in Drm Keys as: /Category/document_id */
  def seq2SparseCategoryParser: CategoryParser = x => x.split("/")(1)



  /**
   * Distributed training of a Naive Bayes model. Follows the approach presented in Rennie et.al.: Tackling the poor
   * assumptions of Naive Bayes Text classifiers, ICML 2003, http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
   *
   * @param observationsPerLabel a DrmLike[Int] matrix containing term frequency counts for each label.
   * @param trainComplementary whether or not to train a complementary Naive Bayes model
   * @param alphaI smoothing parameter
   * @return trained naive bayes model
   */
  def trainNB (observationsPerLabel: DrmLike[Int], trainComplementary: Boolean = true,
    alphaI: Float = defaultAlphaI): NBModel = {

    // Summation of all weights per feature
    val weightsPerFeature = observationsPerLabel.colSums

    // Distributed summation of all weights per label
    val weightsPerLabel = observationsPerLabel.rowSums

    // Collect a matrix to pass to the NaiveBayesModel
    val inCoreTFIDF=observationsPerLabel.collect

    // perLabelThetaNormalizer Vector is expected by NaiveBayesModel. We can pass a null value
    // or Vector of zeroes in the case of a standard NB model.
    var thetaNormalizer= weightsPerFeature.like()

    // Instantiate a trainer and retrieve the perLabelThetaNormalizer Vector from it in the case of
    // a complementary NB model
    if (trainComplementary) {
      val thetaTrainer = new ComplementaryThetaTrainer(weightsPerFeature, weightsPerLabel, alphaI)
      // local training of the theta normalization
      for (labelIndex <- 0 until inCoreTFIDF.nrow) {
        thetaTrainer.train(labelIndex, inCoreTFIDF(labelIndex, ::))
      }
      thetaNormalizer = thetaTrainer.retrievePerLabelThetaNormalizer()
    }

    new NBModel(inCoreTFIDF, weightsPerFeature, weightsPerLabel,
      thetaNormalizer, alphaI, trainComplementary)

  }


  /**
   * Extract label Keys from raw TF or TF-IDF Matrix generated by seq2sparse
   * Override this method in engine specific modules to optimize
   * @param stringKeyedObservations DrmLike matrix; Output from seq2sparse
   *   in form K= eg./Category/document_title
   *           V= TF or TF-IDF values per term
   * @param cParser a String => String function used to extract categories from
   *   Keys of the stringKeyedObservations DRM. The default
   *   CategoryParser will extract "Category" from: '/Category/document_id'
   * @return  (labelIndexMap,aggregatedByLabelObservationDrm)
   *   labelIndexMap is a HashMap  K= label row index
   *                               V= label
   *   aggregatedByLabelObservationDrm is a DrmLike[Int] of aggregated
   *   TF or TF-IDF counts per label
   */
  def extractLabelsAndAggregateObservations[K:ClassTag]
    (stringKeyedObservations: DrmLike[K], cParser: CategoryParser = seq2SparseCategoryParser):
    (mutable.HashMap[String,Double],DrmLike[Int]) = {

    implicit val distributedContext = stringKeyedObservations.context

    stringKeyedObservations.checkpoint()

    val numDocs=stringKeyedObservations.nrow
    val numFeatures=stringKeyedObservations.ncol

    // Extract categories from labels assigned by seq2sparse
    // Categories are Stored in Drm Keys as eg.: /Category/document_id

    // Get a new DRM with a single column so that we don't have to collect the
    // DRM into memory upfront.
    val strippedObeservations= stringKeyedObservations.mapBlock(ncol=1){
      case(keys, block) =>
        val blockB = block.like(keys.size, 1)
        keys -> blockB
    }

    // Extract the row label bindings (the String keys) from the slim Drm
    // strip the document_id from the row keys keeping only the category.
    // Sort the bindings aplhabetically into a Vector
    val labelVectorByRowIndex = strippedObeservations.getRowLabelBindings
                                                       .map(x => x._2 -> cParser(x._1))
                                                       .toVector.sortWith(_._1 < _._1)

    // Copy stringKeyedObservations to an Int-Keyed Drm so that we can compute transpose
    /* Copy the Collected Matrices up front
      val inCoreStringKeyedObservations = stringKeyedObservations.collect
      val inCoreIntKeyedObservations = new SparseMatrix(
                              stringKeyedObservations.nrow.toInt,
                              stringKeyedObservations.ncol)
      for (i <- 0 until inCoreStringKeyedObservations.nrow.toInt) {
        inCoreIntKeyedObservations(i, ::) = inCoreStringKeyedObservations(i, ::)
      }
      val intKeyedObservations= drmParallelize(inCoreIntKeyedObservations)
    */

    // Copy the Distributed Matrices Iterate through and bind one column at a time.
    // Very inefficient, but keeps us from pulling the full dataset upfront.
    var singleColumnInCore= sparse(stringKeyedObservations.collect(::, 0)).t
    var singleColumnDrm=drmParallelize(singleColumnInCore)
    var intKeyedObservations = drmParallelize(singleColumnInCore)
    for (i <- 1 until numFeatures) {
      singleColumnInCore= sparse(stringKeyedObservations.collect(::, i)).t
      singleColumnDrm = drmParallelize(singleColumnInCore)
      intKeyedObservations = intKeyedObservations cbind drmParallelize(singleColumnInCore)
    }

    stringKeyedObservations.uncache()

    var categoryIndex = 0.0d
    val encodedCategoryByKey = new mutable.HashMap[String,Double]
    val encodedCategoryByRowIndexVector = new DenseVector(labelVectorByRowIndex.size)

    // Encode Categories as an Integer (Double) so we can broadcast as a vector
    // where each element is an Int-encoded category whose index corresponds
    // to its row in the Drm
    for (i <- 0 until labelVectorByRowIndex.size) {
      if (!(encodedCategoryByKey.contains(labelVectorByRowIndex(i)._2))) {
        encodedCategoryByRowIndexVector.set(i, categoryIndex)
        encodedCategoryByKey.put(labelVectorByRowIndex(i)._2, categoryIndex)
        categoryIndex += 1.0
      }
      //println(i+" map does not contain: "+labelVectorByRowIndex(i)._2)
      encodedCategoryByRowIndexVector.set(i ,
                            encodedCategoryByKey
                              .getOrElse(labelVectorByRowIndex(i)._2, -1)
                              .asInstanceOf[Double])
    }

    // "Combiner": Map and aggregate by Category. Do this by broadcasting the encoded
    // category vector and mapping a transposed IntKeyed Drm out so that all categories
    // will be present on all nodes as columns and can be referenced by
    // BCastEncodedCategoryByRowVector.  Iteratively sum all categories.
    val ncategories = categoryIndex.toInt

    val BCastEncodedCategoryByRowVector= drmBroadcast(encodedCategoryByRowIndexVector)

    val aggregetedObservationByLabelDrm = intKeyedObservations.t.mapBlock(ncol = ncategories) {
      case (keys, blockA) =>
        val blockB = blockA.like(keys.size, ncategories)
        var category : Int = 0
        for (i <- 0 until keys.size) {
          blockA(i, ::).nonZeroes().foreach { elem =>
            category = BCastEncodedCategoryByRowVector.get(elem.index).toInt
            blockB(i, category) = blockB(i, category) + blockA(i, elem.index)
          }
        }
        keys -> blockB
    }.t

    // Now return the labelMapByRowIndex HashMap and the the
    // aggregetedObservationDrm which can be used as input to trainNB
    (encodedCategoryByKey, aggregetedObservationByLabelDrm)
  }


}

