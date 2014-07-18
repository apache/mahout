/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.sparkbindings.drm.classification

import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel
import org.apache.mahout.classifier.naivebayes.training.ComplementaryThetaTrainer
import scala.reflect.ClassTag

/**
 * Distributed training of a Naive Bayes model. Follows the approach presented in Rennie et.al.: Tackling the poor
 * assumptions of Naive Bayes Text classifiers, ICML 2003, http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
 */
object NaiveBayes {

  /** default value for the smoothing parameter */
  def defaultAlphaI = 1f

  /**
   * Distributed training of a Naive Bayes model.
   *
   * @param observationsPerLabel an array of matrices. Every matrix contains the observations for a particular label.
   * @param trainComplementary whether to train a complementary Naive Bayes model
   * @param alphaI smoothing parameter
   * @return trained naive bayes model
   */
  def trainNB[K: ClassTag](observationsPerLabel: Array[DrmLike[K]],
                                   alphaI: Float = defaultAlphaI): NaiveBayesModel = {

    // distributed summation of all observations per label
    val weightsPerLabelAndFeature = scalabindings.dense(observationsPerLabel.map(new MatrixOps(_).colSums))
    // local summation of all weights per feature
    val weightsPerFeature = new MatrixOps(weightsPerLabelAndFeature).colSums
    // local summation of all weights per label
    val weightsPerLabel = new MatrixOps(weightsPerLabelAndFeature).rowSums

    // instantiate a trainer for the theta normalization
    val thetaTrainer = new ComplementaryThetaTrainer(weightsPerFeature, weightsPerLabel, alphaI)
    // local training of the theta normalization
    for (labelIndex <- 0 until new MatrixOps(weightsPerLabelAndFeature).nrow) {
      thetaTrainer.train(labelIndex, weightsPerLabelAndFeature.viewRow(labelIndex))
    }

    new NaiveBayesModel(weightsPerLabelAndFeature, weightsPerFeature, weightsPerLabel,
                        thetaTrainer.retrievePerLabelThetaNormalizer(), alphaI, true)
  }
}
