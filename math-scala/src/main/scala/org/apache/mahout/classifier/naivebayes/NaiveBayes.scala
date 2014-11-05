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

import java.util

import org.apache.mahout.math._
import org.apache.mahout.math.function.VectorFunction
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

  /** default value for the smoothing parameter */
  def defaultAlphaI = 1.0f

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
    alphaI: Float = defaultAlphaI): NaiveBayesModel = {

    // Summation of all weights per feature
    val weightsPerFeature = observationsPerLabel.colSums

    // distributed summation of all weights per label
    val weightsPerLabel = observationsPerLabel.rowSums

    // collect a matrix to pass to the NaiveBayesModel
    // todo:see if there is anything keeping us from passing a DRM through the rest of the pipeline
    val inCoreTFIDF=observationsPerLabel.collect

    // perLabelThetaNormalizer Vector is expected by NaiveBayesModel. We can pass a null value
    // or Vector of zeroes in the case of a standard NB model.
    var thetaNormalizer= weightsPerFeature.like()

    // instantiate a trainer and retrieve the perLabelThetaNormalizer Vector from it in the case of
    // a complementary NB model
    if (trainComplementary) {
      val thetaTrainer = new ComplementaryThetaTrainer(weightsPerFeature, weightsPerLabel, alphaI)
      // local training of the theta normalization
      for (labelIndex <- 0 until inCoreTFIDF.nrow) {
        thetaTrainer.train(labelIndex, inCoreTFIDF(labelIndex, ::))
      }
      thetaNormalizer = thetaTrainer.retrievePerLabelThetaNormalizer()
    }

    // todo: need a new NaiveBayesModel?
    new NaiveBayesModel(inCoreTFIDF, weightsPerFeature, weightsPerLabel,
      thetaNormalizer, alphaI, trainComplementary)

  }

  /** extract label Keys from raw TF or TF-IDF Matrix generated by seq2sparse
    * Override this method in engine specific modules to optimize
    *
    * @param stringKeyedObservations DrmLike matrix; Output from seq2sparse
    *                                in form K= /Category/document_title
    *                                        V= TF or TF-IDF values per term
    * @return (labelIndexMap,aggregatedByLabelObservationDrm)
    *
    *           labelIndexMap is an HashMap  K= label index
    *                                        V= label
    *           aggregatedByLabelObservationDrm is a DrmLike[Int] of aggregated
    *             TF or TF-IDF counts per label
    */
  def extractLabelsAndAggregateObservations[K:ClassTag]( stringKeyedObservations: DrmLike[K] ):
  (mutable.HashMap[Integer, String], DrmLike[Int]) = {

    implicit val distributedContext = stringKeyedObservations.context

    // Extract categories from labels assigned by seq2sparse
    // Categories are Stored in Drm Keys as: /Category/document_id

    // get a new DRM with a single column so that we don't have to collect the
    // DRM into memory upfont
    val strippedObeservations= stringKeyedObservations.mapBlock(ncol=1){
      case(keys, block) =>
        val blockB = block.like(keys.size, 1)
        keys -> blockB
    }
    val rowLabelBindings= strippedObeservations.getRowLabelBindings
    // sort the bindings into a list
    val labelListSorted = rowLabelBindings.toList.sortWith(_._2 < _._2)
    // strip the document_id from the row keys keeping only the category
    val labelMapByRowIndex = labelListSorted.toMap.map(x => x._2 -> x._1.split("/")(1))

//    for ((key, value) <- rowLabelBindings) {
//      println(value)
//      labelVectorByRowIndex(value)= key.split("/")(1)
//      labelMapByRowIndex.put(value, key.split("/")(1))
//    }

    // convert to an IntKeyed Drm so that we can compute transpose
    // must be a better way to do this.
    // todo: use slice to copy DRM over
    // todo: if doing this iteratively
    val intKeyedObservations = drmParallelizeEmpty(
                            stringKeyedObservations.nrow.toInt,
                            stringKeyedObservations.ncol)
    for (i <- 0 until stringKeyedObservations.nrow.toInt) {
      for ( j <- 0 until stringKeyedObservations.ncol) {
        intKeyedObservations.set(i, j,
                                 stringKeyedObservations.get(i,j))
      }
    }

    // get rid of stringKeyedObservations - we don't need them anymore
    // how do we "free" them?- I know uncache is incorrect.
   // stringKeyedObservations.uncache

    var categoryIndex = 0.0d
    val encodedCategoryByKey = new mutable.HashMap[String,Integer]
    val encodedCategoryByRowIndexVector = new DenseVector(labelVectorByRowIndex.size)

    // encode Categories as a (Double)Integer so we can broadcast as a vector
    // where each element is an Int-encoded category whose index corresponds
    // to its row in the Drm
    for (i <- 0 until labelVectorByRowIndex.size) {
      if (!encodedCategoryByKey.contains()) {
        encodedCategoryByRowIndexVector.set(i, categoryIndex)
        encodedCategoryByKey.put(labelVectorByRowIndex.get(i), i)
        categoryIndex += 1.0
      } else {
        encodedCategoryByRowIndexVector.set(i ,
                              encodedCategoryByKey
                                .getOrElse(labelVectorByRowIndex.get(i), -1)
                                .asInstanceOf[Double])
      }
    }

    // "Combiner": Map and aggregate by Category. Do this by
    // broadcasting the encoded category vector and mapping
    // a transposed IntKeyed Drm out so that all categories
    // will be present on all nodes as columns and can be referenced
    // by BCastEncodedCategoryByRowVector.  Iteratively sum all categories
    val ncategories = categoryIndex.toInt

    val BCastEncodedCategoryByRowVector= drmBroadcast(encodedCategoryByRowIndexVector)

    val aggregetedObservationByLabelDrm = intKeyedObservations.t.mapBlock(ncol = ncategories) {
      case (keys, blockA) =>
        val blockB = blockA.like(keys.size, ncategories)
        //val blockB = new SparseRowMatrix((keys.size, ncategories)
        var category : Int = 0
        for (i <- 0 until keys.size) {
          // todo: Should probably use nonZeroes here as well
          for (j <- 0 until blockA.ncol) {
            category = BCastEncodedCategoryByRowVector.get(j).toInt
            blockB.setQuick(i, category, (blockB.get(i,category) + blockA.get(i,j)))
          }
        }
        keys -> blockB
    }

    // get rid of intKeyedObservations- we don't need them any more
    // how do we "free" them?- I know uncache is incorrect.
    intKeyedObservations.uncache

    // Now return the labelMapByRowIndex HashMap and the the transpose of
    // aggregetedObservationDrm which can be used as input to trainNB
    (labelMapByRowIndex, aggregetedObservationByLabelDrm.t)
  }

}

