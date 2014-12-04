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
abstract class AbstractNBClassifier(nbModel: NBModel) {
  val model = nbModel

  protected def getScoreForLabelFeature(label: Int, feature: Int): Double

  protected def getModel: NBModel= {
     model
  }

  protected def getScoreForLabelInstance(label: Int, instance: Vector): Double = {
    var result: Double = 0.0
    for (e <- instance.nonZeroes) {
      result += e.get * getScoreForLabelFeature(label, e.index)
    }
    result
  }

  def numCategories: Int = {
     model.numLabels
  }

  def classifyFull(instance: Vector): Vector = {
    return classifyFull(model.createScoringVector, instance)
  }

  def classifyFull(r: Vector, instance: Vector): Vector = {
      var label: Int = 0
    for (label <- 0 until model.numLabels) {
      r.setQuick(label, getScoreForLabelInstance(label, instance))
      }
    r
  }
}

/**
 * Standard Classifier
 * @param nbModel a trained NBModel
 */
class StandardNBClassifier(nbModel: NBModel) extends AbstractNBClassifier(nbModel: NBModel) {
  def getScoreForLabelFeature(label: Int, feature: Int): Double = {
    val model: NBModel = getModel
    StandardNBClassifier.computeWeight(model.weight(label, feature), model.labelWeight(label), model.alphaI, model.numFeatures)
  }
}

object StandardNBClassifier  {
  /**
   *
   * @param featureLabelWeight
   * @param labelWeight
   * @param alphaI
   * @param numFeatures
   * @return
   */
  def computeWeight(featureLabelWeight: Double, labelWeight: Double, alphaI: Double, numFeatures: Double): Double = {
    val numerator: Double = featureLabelWeight + alphaI
    val denominator: Double = labelWeight + alphaI * numFeatures
    return Math.log(numerator / denominator)
  }
}

/**
 * Complemtary Classifier
 * @param nbModel a trained NBModel
 */
class ComplementaryNBClassifier(nbModel: NBModel) extends AbstractNBClassifier(nbModel: NBModel) {

  def getScoreForLabelFeature(label: Int, feature: Int): Double = {
    val model: NBModel = getModel
    val weight: Double = ComplementaryNBClassifier.computeWeight(model.featureWeight(feature), model.weight(label, feature), model.totalWeightSum, model.labelWeight(label), model.alphaI, model.numFeatures)
    return weight / model.thetaNormalizer(label)
  }
}

object ComplementaryNBClassifier  {
  /**
   * Calculate weight normalized complementary score
   * @param featureWeight
   * @param featureLabelWeight
   * @param totalWeight
   * @param labelWeight
   * @param alphaI
   * @param numFeatures
   * @return
   */
  def computeWeight(featureWeight: Double, featureLabelWeight: Double, totalWeight: Double, labelWeight: Double, alphaI: Double, numFeatures: Double): Double = {
    val numerator: Double = featureWeight - featureLabelWeight + alphaI
    val denominator: Double = totalWeight - labelWeight + alphaI * numFeatures
    return -Math.log(numerator / denominator)
  }
}
