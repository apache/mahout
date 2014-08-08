package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.math._
import org.apache.mahout.math.function.VectorFunction
import scalabindings._
import scalabindings.RLikeOps._
import drm.RLikeDrmOps._
import drm._
import scala.reflect.ClassTag

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
   * @param observationsPerLabel a DrmLike matrix containing term frequency counts for each label.
   * @param trainComplementary whether or not to train a complementary Naive Bayes model
   * @param alphaI smoothing parameter
   * @return trained naive bayes model
   */
  def trainNB[K: ClassTag](observationsPerLabel: DrmLike[K], trainComplementary: Boolean = true,
                           alphaI: Float = defaultAlphaI): NaiveBayesModel = {

    // Summation of all weights per feature
    val weightsPerFeature = observationsPerLabel.colSums()

    // local summation of all weights per label
    /** Since this is not distributed might as well wait until after collecting **/
    /** val weightsPerLabel2 = observationsPerLabel.aggregateRows(vectorSumFunc)**/
    /*****************************************************************************/

    // collect a matrix to pass to the NaiveBayesModel
    val inCoreTFIDF=observationsPerLabel.collect

    // local summation of all weights per label
    val weightsPerLabel = inCoreTFIDF.rowSums()

    // perLabelThetaNormalizer Vector is expected by NaiveBayesModel. We can pass a null value
    // or Vector of zeroes in the case of a standard NB model.
    var thetaNormalizer= weightsPerFeature.like()

    // instantiate a trainer and retrieve the perLabelThetaNormalizer Vector from it in the case of
    // a complementary NB model
    if (trainComplementary) {
      val thetaTrainer = new ComplementaryThetaTrainer(weightsPerFeature, weightsPerLabel, alphaI)
      // local training of the theta normalization
      for (labelIndex <- 0 until inCoreTFIDF.nrow.toInt) {
        thetaTrainer.train(labelIndex, inCoreTFIDF(labelIndex, ::))
      }
      thetaNormalizer = thetaTrainer.retrievePerLabelThetaNormalizer()
    }

    new NaiveBayesModel(inCoreTFIDF, weightsPerFeature, weightsPerLabel,
      thetaNormalizer, alphaI, trainComplementary)

  }
//  private def vectorSumFunc = new VectorFunction {
//    def apply(f: Vector): Double = f.sum
//  }

}

