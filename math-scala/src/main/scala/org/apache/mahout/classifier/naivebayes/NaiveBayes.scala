package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.classifier.naivebayes.training.ComplementaryThetaTrainer
import org.apache.mahout.math._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import scala.reflect.ClassTag

/**
 * Distributed training of a Naive Bayes model. Follows the approach presented in Rennie et.al.: Tackling the poor
 * assumptions of Naive Bayes Text classifiers, ICML 2003, http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
 */
object NaiveBayes {

  /** default value for the smoothing parameter */
  def defaultAlphaI = 1.0

  /**
   * Distributed training of a Naive Bayes model. Follows the approach presented in Rennie et.al.: Tackling the poor
   * assumptions of Naive Bayes Text classifiers, ICML 2003, http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
   *
   * @param observationsPerLabel an array of matrices. Every matrix contains the observations for a particular label.
   * @param trainComplementary whether to train a complementary Naive Bayes model
   * @param alphaI smoothing parameter
   * @return trained naive bayes model
   */
  def trainNB[K: ClassTag](observationsPerLabel: Array[DrmLike[K]], trainComplementary: Boolean = true,
                           alphaI: Double = defaultAlphaI): NaiveBayesModel = {


    // distributed summation of all observations per label
    val weightsPerLabelAndFeature = dense(observationsPerLabel.map(_.colSums))
    // local summation of all weights per feature
    val weightsPerFeature = weightsPerLabelAndFeature.colSums
    // local summation of all weights per label
    val weightsPerLabel = weightsPerLabelAndFeature.rowSums

    // perLabelThetaNormalizer Vector is expected by NaiveBayesModel. We can pass a null value
    // in the case of a standard NB model
    var thetaNormalizer: Vector = null


    // instantiate a trainer and retrieve the perLabelThetaNormalizer Vector from it in the case of
    // a complementary NB model
    if (trainComplementary) {
      val thetaTrainer = new ComplementaryThetaTrainer(weightsPerFeature, weightsPerLabel, alphaI)
      // local training of the theta normalization
      for (labelIndex <- 0 until weightsPerLabelAndFeature.nrow) {
        thetaTrainer.train(labelIndex, weightsPerLabelAndFeature(labelIndex, ::))
      }
      thetaNormalizer = thetaTrainer.retrievePerLabelThetaNormalizer()
    }

    new org.apache.mahout.classifier.naivebayes.NaiveBayesModel(weightsPerLabelAndFeature, weightsPerFeature, weightsPerLabel,
      thetaNormalizer, alphaI, trainComplementary)

  }

}

