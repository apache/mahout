/**
  * Created by saikat on 6/6/16.
  */
package org.apache.mahout.perf
import scala.concurrent.duration._
import org.apache.mahout.classifier.naivebayes.NaiveBayes
import org.apache.mahout.math.drm
import org.apache.mahout.math.scalabindings._

trait PerfMeasurementDriver  {
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

  val cat1=labelIndex("Cat1")
  val cat2=labelIndex("Cat2")

  cat1 should be (0)
  cat2 should be (1)

  val aggregatedTFIDFInCore = aggregatedTFIDFDrm.collect

}