package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}


trait NBTestBase extends DistributedMahoutSuite with Matchers { this:FunSuite =>

  test("Simple Standard NB Model") {

    val epsilon = 1E-4 //keeping wide threshold for tonight

    // test from simulated sparse TF-IDF data
    val inCoreTFIDF = sparse(
      (0, 0.7) ::(1, 0.1) ::(2, 0.1) ::(3, 0.3) :: Nil,
      (0, 0.4) ::(1, 0.4) ::(2, 0.1) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.8) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.1) ::(2, 0.1) ::(3, 0.7) :: Nil
    )

    val TFIDFDrm = drm.drmParallelize(m = inCoreTFIDF, numPartitions = 2)

    // train a Standard NB Model
    val model = NaiveBayes.trainNB(TFIDFDrm, false)

    // validate the model- will throw an exception if model is invalid
    model.validate()

    // check the labelWeights
    model.labelWeight(0) - 1.2 should be < epsilon
    model.labelWeight(1) - 1.0 should be < epsilon
    model.labelWeight(2) - 1.0 should be < epsilon
    model.labelWeight(3) - 1.0 should be < epsilon

    // check the Feature weights
    model.featureWeight(0) - 1.3 should be < epsilon
    model.featureWeight(1) - 0.6 should be < epsilon
    model.featureWeight(2) - 1.1 should be < epsilon
    model.featureWeight(3) - 1.2 should be < epsilon
  }

}
