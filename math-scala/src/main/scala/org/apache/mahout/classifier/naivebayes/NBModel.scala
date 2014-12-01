package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.math._

import org.apache.mahout.math.{drm, scalabindings}

import scalabindings._
import scalabindings.RLikeOps._
import drm.RLikeDrmOps._
import drm._
import scala.language.asInstanceOf
import collection._
import JavaConversions._



class NBModel {
  var weightsPerLabel: Vector = null
  var perlabelThetaNormalizer: Vector = null
  var weightsPerFeature: Vector = null
  var weightsPerLabelAndFeature: Matrix = null
  var alphaI: Float = .0f
  var numFeatures: Double = 0.0
  var totalWeightSum: Double = 0.0
  var isComplementary: Boolean = false
  var alphaVector: Vector= null

  // todo: is distributed context being set up correctly here?  Maybe it is a good
  // idea to move the serialize and materialize out of the model and into a helper

  //implicit var distributedContext: DistributedContext

  /**
   *
   * @param weightMatrix Aggregated matrix of weights of features x labels
   * @param featureWeights Vector of summation of all feature weights.
   * @param labelWeights Vector of summation of all label weights.
   * @param weightNormalizers Vector of weight normalizers per label (used only for complemtary models)
   * @param alphaI Laplacian smoothing factor.
   * @param isComplementary Whether or not this is a complementary model.
   */
  def this(weightMatrix: Matrix,
           featureWeights: Vector,
           labelWeights: Vector,
           weightNormalizers: Vector,
           alphaI: Float,
           isComplementary: Boolean) {
    this()
    // TODO: weightsPerLabelAndFeature, a sparse (numFeatures x numLabels) matrix should fit
    // TODO: upfront in memory and should not require a DRM decide if we want this to scale out.
    this.weightsPerLabelAndFeature = weightMatrix
    this.weightsPerFeature = featureWeights
    this.weightsPerLabel = labelWeights
    this.perlabelThetaNormalizer = weightNormalizers
    this.numFeatures = featureWeights.getNumNondefaultElements
    this.totalWeightSum = labelWeights.zSum
    this.alphaI = alphaI
    this.isComplementary = isComplementary

  }
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



  /**
   * Write a trained model to the filesystem as a set of DRMs
   * @param pathToModel Directory to which the model will be written
   */
  def serialize(pathToModel: String)(implicit ctx: DistributedContext): Unit = {
    drmParallelize(weightsPerLabelAndFeature).dfsWrite(pathToModel + "/weightsPerLabelAndFeatureDrm.drm")
    drmParallelize(dense(weightsPerFeature)).dfsWrite(pathToModel + "/weightsPerFeatureDrm.drm")
    drmParallelize(dense(weightsPerLabel)).dfsWrite(pathToModel + "/weightsPerLabelDrm.drm")
    drmParallelize(dense(perlabelThetaNormalizer)).dfsWrite(pathToModel + "/perlabelThetaNormalizerDrm.drm")
    drmParallelize(dense(dvec(alphaI))).dfsWrite(pathToModel + "/alphaIDrm.drm")
    // isComplementry is true if isComplementaryDrm(0,0) == 1 else false
    val isComplementaryDrm = dense(1,1)
    if(isComplementary){
      isComplementaryDrm(0,0) = 1.0
    } else {
      isComplementaryDrm(0,0) = 0.0
    }
    drmParallelize(isComplementaryDrm).dfsWrite(pathToModel + "/isComplementaryDrm.drm")

  }

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
      assert(perlabelThetaNormalizer.getNumNonZeroElements == perlabelThetaNormalizer.size, "Theta normalizers can not have zero value.")
    }
  }

}

object NBModel{
  /**
   * Read a trained model in from from the filesystem.
   * @param pathToModel directory from which to read individual model components
   * @return
   */
  def materialize(pathToModel: String)(implicit ctx: DistributedContext): NBModel = {
    val weightsPerLabelAndFeatureDrm = drmDfsRead(pathToModel + "/weightsPerLabelAndFeatureDrm.drm")
    val weightsPerFeatureDrm = drmDfsRead(pathToModel + "/weightsPerFeatureDrm.drm")
    val weightsPerLabelDrm = drmDfsRead(pathToModel + "/weightsPerLabelDrm.drm")
    val alphaIDrm = drmDfsRead(pathToModel + "/alphaIDrm.drm")

    // isComplementry is true if isComplementaryDrm(0,0) == 1 else false
    val isComplementaryDrm = drmDfsRead(pathToModel + "/isComplementaryDrm.drm")

    val weightsPerLabelAndFeature = weightsPerLabelAndFeatureDrm.collect
    val weightsPerFeature = weightsPerFeatureDrm.collect(0, ::)
    val weightsPerLabel = weightsPerLabelDrm.collect(0, ::)

    val alphaI: Float = alphaIDrm.collect(0, 0).toFloat
    val isComplementary = isComplementaryDrm.collect(0, 0).toInt == 1

    var perLabelThetaNormalizer= weightsPerFeature.like()
    if (isComplementary) {
      val perLabelThetaNormalizerDrm = drm.drmDfsRead(pathToModel + "/perlabelThetaNormalizerDrm.drm")
      perLabelThetaNormalizer = perLabelThetaNormalizerDrm.collect(0, ::)
    }

    val model: NBModel = new NBModel(weightsPerLabelAndFeature,
      weightsPerFeature,
      weightsPerLabel,
      perLabelThetaNormalizer,
      alphaI,
      isComplementary)
    model.validate

    model
  }
}
