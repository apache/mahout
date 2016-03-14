/*
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
*/
package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.math.Vector
import scala.collection.JavaConversions._

/**
 * Abstract Classifier base for Complentary and Standard Classifiers
 * @param nbModel a trained NBModel
 */
abstract class AbstractNBClassifier(nbModel: NBModel) extends java.io.Serializable {

  // Trained Naive Bayes Model
  val model = nbModel

  /** scoring method for standard and complementary classifiers */
  protected def getScoreForLabelFeature(label: Int, feature: Int): Double

  /** getter for model */
  protected def getModel: NBModel= {
     model
  }

  /**
   * Compute the score for a Vector of weighted TF-IDF featured
   * @param label Label to be scored
   * @param instance Vector of weights to be calculate score
   * @return score for this Label
   */
  protected def getScoreForLabelInstance(label: Int, instance: Vector): Double = {
    var result: Double = 0.0
    for (e <- instance.nonZeroes) {
      result += e.get * getScoreForLabelFeature(label, e.index)
    }
    result
  }

  /** number of categories the model has been trained on */
  def numCategories: Int = {
     model.numLabels
  }

  /**
   * get a scoring vector for a vector of TF of TF-IDF weights
   * @param instance vector of TF of TF-IDF weights to be classified
   * @return a vector of scores.
   */
  def classifyFull(instance: Vector): Vector = {
    classifyFull(model.createScoringVector, instance)
  }

  /** helper method for classifyFull(Vector) */
  def classifyFull(r: Vector, instance: Vector): Vector = {
    var label: Int = 0
    for (label <- 0 until model.numLabels) {
        r.setQuick(label, getScoreForLabelInstance(label, instance))
      }
    r
  }
}

/**
 * Standard Multinomial Naive Bayes Classifier
 * @param nbModel a trained NBModel
 */
class StandardNBClassifier(nbModel: NBModel) extends AbstractNBClassifier(nbModel: NBModel) with java.io.Serializable{
  override def getScoreForLabelFeature(label: Int, feature: Int): Double = {
    val model: NBModel = getModel
    StandardNBClassifier.computeWeight(model.weight(label, feature), model.labelWeight(label), model.alphaI, model.numFeatures)
  }
}

/** helper object for StandardNBClassifier */
object StandardNBClassifier extends java.io.Serializable {
  /** Compute Standard Multinomial Naive Bayes Weights See Rennie et. al. Section 2.1 */
  def computeWeight(featureLabelWeight: Double, labelWeight: Double, alphaI: Double, numFeatures: Double): Double = {
    val numerator: Double = featureLabelWeight + alphaI
    val denominator: Double = labelWeight + alphaI * numFeatures
    Math.log(numerator / denominator)
  }
}

/**
 * Complementary Naive Bayes Classifier
 * @param nbModel a trained NBModel
 */
class ComplementaryNBClassifier(nbModel: NBModel) extends AbstractNBClassifier(nbModel: NBModel) with java.io.Serializable {
  override def getScoreForLabelFeature(label: Int, feature: Int): Double = {
    val model: NBModel = getModel
    val weight: Double = ComplementaryNBClassifier.computeWeight(model.featureWeight(feature), model.weight(label, feature), model.totalWeightSum, model.labelWeight(label), model.alphaI, model.numFeatures)
    weight / model.thetaNormalizer(label)
  }
}

/** helper object for ComplementaryNBClassifier */
object ComplementaryNBClassifier extends java.io.Serializable {

  /** Compute Complementary weights See Rennie et. al. Section 3.1 */
  def computeWeight(featureWeight: Double, featureLabelWeight: Double, totalWeight: Double, labelWeight: Double, alphaI: Double, numFeatures: Double): Double = {
    val numerator: Double = featureWeight - featureLabelWeight + alphaI
    val denominator: Double = totalWeight - labelWeight + alphaI * numFeatures
    -Math.log(numerator / denominator)
  }
}
