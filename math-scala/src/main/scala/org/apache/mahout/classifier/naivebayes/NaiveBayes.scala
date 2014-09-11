package org.apache.mahout.classifier.naivebayes

import java.util

import org.apache.mahout.math._
import org.apache.mahout.math.function.VectorFunction
import org.apache.mahout.math.map.{OpenIntObjectHashMap, OpenObjectIntHashMap}
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
   * @param observationsPerLabel a DrmLike matrix containing term frequency counts for each label.
   * @param trainComplementary whether or not to train a complementary Naive Bayes model
   * @param alphaI smoothing parameter
   * @return trained naive bayes model
   */
  def trainNB[K: ClassTag](observationsPerLabel: DrmLike[K], trainComplementary: Boolean = true,
    alphaI: Float = defaultAlphaI): NaiveBayesModel = {

    // Summation of all weights per feature
    val weightsPerFeature = observationsPerLabel.colSums

    // local summation of all weights per label
    /** Since this is not distributed might as well wait until after collecting **/
    /** val weightsPerLabel2 = observationsPerLabel.aggregateRows(vectorSumFunc)**/
    /*****************************************************************************/

    // collect a matrix to pass to the NaiveBayesModel
    val inCoreTFIDF=observationsPerLabel.collect

    // local summation of all weights per label
    val weightsPerLabel = inCoreTFIDF.rowSums

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

    new NaiveBayesModel(inCoreTFIDF, weightsPerFeature, weightsPerLabel,
      thetaNormalizer, alphaI, trainComplementary)

  }

  /** extract label Keys from raw TF/TF-IDF Matrix
    *
    * @param stringKeyedObservations DrmLike matrix; Output from seq2sparse
    *                                in form K= /Category/document_title
    *                                        V= TF or TF-IDF values per term
    * @return (labelIndexMap,aggregatedByLabelObservationDrm)
    *           labelIndexMap is an OpenObjectIntHashMap K= label index
    *                                                  V= label
    *           aggregatedByLabelObservationDrm is a DrmLike[Int] of aggregated
    *             TF or TF-IDF counts per label
    */
  def extractLabelsAndAggregateObservations( stringKeyedObservations: DrmLike[String] ):
  (mutable.HashMap[Integer,String], DrmLike[Int]) = {

    implicit val distributedContext = stringKeyedObservations.context

    // get the label row keys as
    val rowLabelBindings = stringKeyedObservations.getRowLabelBindings

    // extract categories from labels assigned by seq2sparse
    // Categories are Stored in Drm Keys as: /Category/document_id
    val labelVectorByRowIndex = new util.Vector[String](rowLabelBindings.size())
    val labelMapByRowIndex = new mutable.HashMap[Integer,String]
    for ((key, value) <- rowLabelBindings) {
      labelVectorByRowIndex.set(value, key.split("/")(1))
      labelMapByRowIndex.put(value, key.split("/")(1) )
    }

    // convert to an IntKeyed Drm
    // must be a better way to do this.
    // XXX: at least use iterateNonZeroes or somethin similar
    // if doing this iteratively
    val intKeyedObservations = drmParallelizeEmpty(
                            stringKeyedObservations.nrow.toInt,
                            stringKeyedObservations.ncol)
    for (i <- 0 until stringKeyedObservations.nrow.toInt) {
      for ( j <- 0 until stringKeyedObservations.ncol) {
        intKeyedObservations.set(i,
                                 j,
                                 stringKeyedObservations.get(i,j))
      }
    }

    // get rid of stringKeyedObservations - we dont need them anymore
    stringKeyedObservations.uncache

    var categoryIndex = 0.0d
    val encodedCategoryByKey = new mutable.HashMap[String,Integer]
    val encodedCategoryByRowIndexVector = new DenseVector(labelVectorByRowIndex.size)

    // encode rows as an (Double)Integer so we can broadcast as a vector
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
        val blockB = new DenseMatrix(keys.size, ncategories)
        //val blockB = blockA.zeroes(keys.size, ncategories)
        var category : Int = 0

        for (i <- 0 until keys.size) {
          for (j <- 0 until blockA.ncol) {
            category = BCastEncodedCategoryByRowVector.get(j).toInt
            blockB.set(i, category, (blockB.get(i,category) + blockA.get(i,j)))
          }
        }
        keys -> blockB
    }

    // get rid of intKeyedObservations we dont need them any more
    intKeyedObservations.uncache

    // Now return the labelMapByRowIndex HashMap and the the transpose of
    // aggregetedObservationDrm which can be used as input to trainNB
    (labelMapByRowIndex, aggregetedObservationByLabelDrm.t)
  }


//  private def vectorSumFunc = new VectorFunction {
//    def apply(f: Vector): Double = f.sum
//  }

}

