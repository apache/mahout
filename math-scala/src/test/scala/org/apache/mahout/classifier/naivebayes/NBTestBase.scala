package org.apache.mahout.classifier.naivebayes

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}
import collection._
import JavaConversions._

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

  test("NB Aggregator") {

    val epsilon = 1E-4 //keeping wide threshold for tonight

    val rowBindings = new java.util.HashMap[String,Integer]()
    rowBindings.put("/Cat1/doc_a/", 0)
    rowBindings.put("/Cat2/doc_b/", 1)
    rowBindings.put("/Cat1/doc_c/", 2)
    rowBindings.put("/Cat2/doc_d/", 3)
    rowBindings.put("/Cat1/doc_e/", 4)


    val matrixSetup = sparse(
      (0, 0.1) ::(1, 0.0) ::(2, 0.1) ::(3, 0.0) :: Nil,
      (0, 0.0) ::(1, 0.1) ::(2, 0.0) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.1) ::(3, 0.0) :: Nil,
      (0, 0.0) ::(1, 0.1) ::(2, 0.0) ::(3, 0.1) :: Nil,
      (0, 0.1) ::(1, 0.0) ::(2, 0.1) ::(3, 0.0) :: Nil
    )


    matrixSetup.setRowLabelBindings(rowBindings)

    val TFIDFDrm = drm.drmParallelizeWithRowLabels(m = matrixSetup, numPartitions = 2)

    val (labelIndex, aggregatedTFIDFDrm) = NaiveBayes.extractLabelsAndAggregateObservations(TFIDFDrm)

    labelIndex.size should be (2)

    val cat1=labelIndex("Cat1").toInt
    val cat2=labelIndex("Cat2").toInt

    cat1 should be (0)
    cat2 should be (1)

    val aggregatedTFIDFInCore = aggregatedTFIDFDrm.collect
    aggregatedTFIDFInCore.numCols should be (4)
    aggregatedTFIDFInCore.numRows should be (2)

    aggregatedTFIDFInCore.get(cat1, 0) - 0.3 should be < epsilon
    aggregatedTFIDFInCore.get(cat1, 1) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat1, 2) - 0.3 should be < epsilon
    aggregatedTFIDFInCore.get(cat1, 3) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 0) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 1) - 0.2 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 2) - 0.0 should be < epsilon
    aggregatedTFIDFInCore.get(cat2, 3) - 0.2 should be < epsilon

  }

}
