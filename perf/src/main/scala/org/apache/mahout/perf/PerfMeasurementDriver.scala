/**
  * Created by saikat on 6/6/16.
  */
package org.apache.mahout.perf
import org.apache.mahout.classifier.naivebayes.NaiveBayes
import org.apache.mahout.math._
import org.apache.mahout.math.drm.{DistributedContext, DistributedEngine}
import org.apache.mahout.math.scalabindings._

import collection._

object PerfMeasurementDriver extends App with PerfDistributedContext{

  def doWork() {
    val rowBindings = new java.util.HashMap[String, Integer]()
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

    val aggregatedTFIDFInCore = aggregatedTFIDFDrm.collect
  }

}