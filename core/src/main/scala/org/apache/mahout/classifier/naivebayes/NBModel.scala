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

import org.apache.mahout.math.{drm, scalabindings}

import scalabindings._
import scalabindings.RLikeOps._
import drm._
import scala.language.asInstanceOf
import scala.collection._
import JavaConversions._

/**
 *
 * @param weightsPerLabelAndFeature Aggregated matrix of weights of labels x features
 * @param weightsPerFeature Vector of summation of all feature weights.
 * @param weightsPerLabel Vector of summation of all label weights.
 * @param perlabelThetaNormalizer Vector of weight normalizers per label (used only for complemtary models)
 * @param labelIndex HashMap of labels and their corresponding row in the weightMatrix
 * @param alphaI Laplace smoothing factor.
 * @param isComplementary Whether or not this is a complementary model.
 */
class NBModel(val weightsPerLabelAndFeature: Matrix = null,
              val weightsPerFeature: Vector = null,
              val weightsPerLabel: Vector = null,
              val perlabelThetaNormalizer: Vector = null,
              val labelIndex: Map[String, Integer] = null,
              val alphaI: Float = 1.0f,
              val isComplementary: Boolean= false)  extends java.io.Serializable {


  val numFeatures: Double = weightsPerFeature.getNumNondefaultElements
  val totalWeightSum: Double = weightsPerLabel.zSum
  val alphaVector: Vector = null

  validate()

  // todo: Maybe it is a good idea to move the dfsWrite and dfsRead out
  // todo: of the model and into a helper

  // TODO: weightsPerLabelAndFeature, a sparse (numFeatures x numLabels) matrix should fit
  // TODO: upfront in memory and should not require a DRM decide if we want this to scale out.


  /** getter for summed label weights.  Used by legacy classifier */
  def labelWeight(label: Int): Double = {
     weightsPerLabel.getQuick(label)
  }

  /** getter for weight normalizers.  Used by legacy classifier */
  def thetaNormalizer(label: Int): Double = {
    perlabelThetaNormalizer.get(label)
  }

  /** getter for summed feature weights.  Used by legacy classifier */
  def featureWeight(feature: Int): Double = {
    weightsPerFeature.getQuick(feature)
  }

  /** getter for individual aggregated weights.  Used by legacy classifier */
  def weight(label: Int, feature: Int): Double = {
    weightsPerLabelAndFeature.getQuick(label, feature)
  }

  /** getter for a single empty vector of weights */
  def createScoringVector: Vector = {
     weightsPerLabel.like
  }

  /** getter for a the number of labels to consider */
  def numLabels: Int = {
     weightsPerLabel.size
  }

  /**
   * Write a trained model to the filesystem as a series of DRMs
   * @param pathToModel Directory to which the model will be written
   */
  def dfsWrite(pathToModel: String)(implicit ctx: DistributedContext): Unit = {
    //todo:  write out as smaller partitions or possibly use reader and writers to
    //todo:  write something other than a DRM for label Index, is Complementary, alphaI.

    // add a directory to put all of the DRMs in
    val fullPathToModel = pathToModel + NBModel.modelBaseDirectory

    drmParallelize(weightsPerLabelAndFeature).dfsWrite(fullPathToModel + "/weightsPerLabelAndFeatureDrm.drm")
    drmParallelize(sparse(weightsPerFeature)).dfsWrite(fullPathToModel + "/weightsPerFeatureDrm.drm")
    drmParallelize(sparse(weightsPerLabel)).dfsWrite(fullPathToModel + "/weightsPerLabelDrm.drm")
    drmParallelize(sparse(perlabelThetaNormalizer)).dfsWrite(fullPathToModel + "/perlabelThetaNormalizerDrm.drm")
    drmParallelize(sparse(svec((0,alphaI)::Nil))).dfsWrite(fullPathToModel + "/alphaIDrm.drm")

    // isComplementry is true if isComplementaryDrm(0,0) == 1 else false
    val isComplementaryDrm = sparse(0 to 1, 0 to 1)
    if(isComplementary){
      isComplementaryDrm(0,0) = 1.0
    } else {
      isComplementaryDrm(0,0) = 0.0
    }
    drmParallelize(isComplementaryDrm).dfsWrite(fullPathToModel + "/isComplementaryDrm.drm")

    // write the label index as a String-Keyed DRM.
    val labelIndexDummyDrm = weightsPerLabelAndFeature.like()
    labelIndexDummyDrm.setRowLabelBindings(labelIndex)
    // get a reverse map of [Integer, String] and set the value of firsr column of the drm
    // to the corresponding row number for it's Label (the rows may not be read back in the same order)
    val revMap = labelIndex.map(x => x._2 -> x._1)
    for(i <- 0 until labelIndexDummyDrm.numRows() ){
      labelIndexDummyDrm.set(labelIndex(revMap(i)), 0, i.toDouble)
    }

    drmParallelizeWithRowLabels(labelIndexDummyDrm).dfsWrite(fullPathToModel + "/labelIndex.drm")
  }

  /** Model Validation */
  def validate() {
    assert(alphaI > 0, "alphaI has to be greater than 0!")
    assert(numFeatures > 0, "the vocab count has to be greater than 0!")
    assert(totalWeightSum > 0, "the totalWeightSum has to be greater than 0!")
    assert(weightsPerLabel != null, "the number of labels has to be defined!")
    assert(weightsPerLabel.getNumNondefaultElements > 0, "the number of labels has to be greater than 0!")
    assert(weightsPerFeature != null, "the feature sums have to be defined")
    assert(weightsPerFeature.getNumNondefaultElements > 0, "the feature sums have to be greater than 0!")
    if (isComplementary) {
      assert(perlabelThetaNormalizer != null, "the theta normalizers have to be defined")
      assert(perlabelThetaNormalizer.getNumNondefaultElements > 0, "the number of theta normalizers has to be greater than 0!")
      assert(Math.signum(perlabelThetaNormalizer.minValue) == Math.signum(perlabelThetaNormalizer.maxValue), "Theta normalizers do not all have the same sign")
      assert(perlabelThetaNormalizer.getNumNonZeroElements == perlabelThetaNormalizer.size, "Weight normalizers can not have zero value.")
    }
    assert(labelIndex.size == weightsPerLabel.getNumNondefaultElements, "label index must have entries for all labels")
  }
}

object NBModel extends java.io.Serializable {

  val modelBaseDirectory = "/naiveBayesModel"

  /**
   * Read a trained model in from from the filesystem.
   * @param pathToModel directory from which to read individual model components
   * @return a valid NBModel
   */
  def dfsRead(pathToModel: String)(implicit ctx: DistributedContext): NBModel = {
    //todo:  Takes forever to read we need a more practical method of writing models. Readers/Writers?

    // read from a base directory for all drms
    val fullPathToModel = pathToModel + modelBaseDirectory

    val weightsPerFeatureDrm = drmDfsRead(fullPathToModel + "/weightsPerFeatureDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val weightsPerFeature = weightsPerFeatureDrm.collect(0, ::)
    weightsPerFeatureDrm.uncache()

    val weightsPerLabelDrm = drmDfsRead(fullPathToModel + "/weightsPerLabelDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val weightsPerLabel = weightsPerLabelDrm.collect(0, ::)
    weightsPerLabelDrm.uncache()

    val alphaIDrm = drmDfsRead(fullPathToModel + "/alphaIDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val alphaI: Float = alphaIDrm.collect(0, 0).toFloat
    alphaIDrm.uncache()

    // isComplementry is true if isComplementaryDrm(0,0) == 1 else false
    val isComplementaryDrm = drmDfsRead(fullPathToModel + "/isComplementaryDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val isComplementary = isComplementaryDrm.collect(0, 0).toInt == 1
    isComplementaryDrm.uncache()

    var perLabelThetaNormalizer= weightsPerFeature.like()
    if (isComplementary) {
      val perLabelThetaNormalizerDrm = drm.drmDfsRead(fullPathToModel + "/perlabelThetaNormalizerDrm.drm")
                                             .checkpoint(CacheHint.MEMORY_ONLY)
      perLabelThetaNormalizer = perLabelThetaNormalizerDrm.collect(0, ::)
    }

    val dummyLabelDrm= drmDfsRead(fullPathToModel + "/labelIndex.drm")
                         .checkpoint(CacheHint.MEMORY_ONLY)
    val labelIndexMap:java.util.Map[String, Integer] = dummyLabelDrm.getRowLabelBindings
    dummyLabelDrm.uncache()

    // map the labels to the corresponding row numbers of weightsPerFeatureDrm (values in dummyLabelDrm)
    val scalaLabelIndexMap: mutable.Map[String, Integer] =
      labelIndexMap.map(x => x._1 -> dummyLabelDrm.get(labelIndexMap(x._1), 0)
        .toInt
        .asInstanceOf[Integer])

    val weightsPerLabelAndFeatureDrm = drmDfsRead(fullPathToModel + "/weightsPerLabelAndFeatureDrm.drm").checkpoint(CacheHint.MEMORY_ONLY)
    val weightsPerLabelAndFeature = weightsPerLabelAndFeatureDrm.collect
    weightsPerLabelAndFeatureDrm.uncache()

    // model validation is triggered automatically by constructor
    val model: NBModel = new NBModel(weightsPerLabelAndFeature,
      weightsPerFeature,
      weightsPerLabel,
      perLabelThetaNormalizer,
      scalaLabelIndexMap,
      alphaI,
      isComplementary)

    model
  }
}
