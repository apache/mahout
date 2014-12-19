/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.stats

import java.lang.Double
import java.util.Random
import java.util.Arrays

import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.Matrix
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}



trait ClassifierStatsTestBase extends DistributedMahoutSuite with Matchers { this: FunSuite =>

  val epsilon = 1E-6

  val smallEpsilon = 1.0

  // FullRunningAverageAndStdDev tests
  test("testFullRunningAverageAndStdDev") {
    val average: RunningAverageAndStdDev = new FullRunningAverageAndStdDev
    assert(0 == average.getCount)
    assert(true == Double.isNaN(average.getAverage))
    assert(true == Double.isNaN(average.getStandardDeviation))
    average.addDatum(6.0)
    assert(1 == average.getCount)
    assert((6.0 - average.getAverage).abs < epsilon)
    assert(true == Double.isNaN(average.getStandardDeviation))
    average.addDatum(6.0)
    assert(2 == average.getCount)
    assert((6.0 - average.getAverage).abs < epsilon)
    assert((0.0 - average.getStandardDeviation).abs < epsilon)
    average.removeDatum(6.0)
    assert(1 == average.getCount)
    assert((6.0 - average.getAverage).abs < epsilon)
    assert(true == Double.isNaN(average.getStandardDeviation))
    average.addDatum(-4.0)
    assert(2 == average.getCount)
    assert((1.0 - average.getAverage).abs < epsilon)
    assert(((5.0 * 1.4142135623730951) - average.getStandardDeviation).abs < epsilon)
    average.removeDatum(4.0)
    assert(1 == average.getCount)
    assert((2.0 + average.getAverage).abs < epsilon)
    assert(true == Double.isNaN(average.getStandardDeviation))
  }

  test("testBigFullRunningAverageAndStdDev") {
    val average: RunningAverageAndStdDev = new FullRunningAverageAndStdDev
    RandomUtils.useTestSeed()
    val r: Random = RandomUtils.getRandom

    for (i <- 0 until 100000) {
      average.addDatum(r.nextDouble() * 1000.0)
    }

    assert((500.0 - average.getAverage).abs < smallEpsilon)
    assert(((1000.0 / Math.sqrt(12.0)) - average.getStandardDeviation).abs < smallEpsilon)
  }

  test("testStddevFullRunningAverageAndStdDev") {
    val runningAverage: RunningAverageAndStdDev = new FullRunningAverageAndStdDev
    assert(0 == runningAverage.getCount)
    assert(true == Double.isNaN(runningAverage.getAverage))
    runningAverage.addDatum(1.0)
    assert(1 == runningAverage.getCount)
    assert((1.0 - runningAverage.getAverage).abs < epsilon)
    assert(true == Double.isNaN(runningAverage.getStandardDeviation))
    runningAverage.addDatum(1.0)
    assert(2 == runningAverage.getCount)
    assert((1.0 - runningAverage.getAverage).abs < epsilon)
    assert((0.0 -runningAverage.getStandardDeviation).abs < epsilon)
    runningAverage.addDatum(7.0)
    assert(3 == runningAverage.getCount)
    assert((3.0 - runningAverage.getAverage).abs < epsilon)
    assert((3.464101552963257 - runningAverage.getStandardDeviation).abs < epsilon)
    runningAverage.addDatum(5.0)
    assert(4 == runningAverage.getCount)
    assert((3.5 - runningAverage.getAverage) < epsilon)
    assert((3.0- runningAverage.getStandardDeviation).abs < epsilon)
  }



  // FullRunningAverage tests
  test("testFullRunningAverage"){
    val runningAverage: RunningAverage = new FullRunningAverage
    assert(0 == runningAverage.getCount)
    assert(true == Double.isNaN(runningAverage.getAverage))
    runningAverage.addDatum(1.0)
    assert(1 == runningAverage.getCount)
    assert((1.0 - runningAverage.getAverage).abs < epsilon)
    runningAverage.addDatum(1.0)
    assert(2 == runningAverage.getCount)
    assert((1.0 - runningAverage.getAverage).abs < epsilon)
    runningAverage.addDatum(4.0)
    assert(3 == runningAverage.getCount)
    assert((2.0 - runningAverage.getAverage) < epsilon)
    runningAverage.addDatum(-4.0)
    assert(4 == runningAverage.getCount)
    assert((0.5 - runningAverage.getAverage).abs < epsilon)
    runningAverage.removeDatum(-4.0)
    assert(3 == runningAverage.getCount)
    assert((2.0 - runningAverage.getAverage).abs < epsilon)
    runningAverage.removeDatum(4.0)
    assert(2 == runningAverage.getCount)
    assert((1.0 - runningAverage.getAverage).abs < epsilon)
    runningAverage.changeDatum(0.0)
    assert(2 == runningAverage.getCount)
    assert((1.0 - runningAverage.getAverage).abs < epsilon)
    runningAverage.changeDatum(2.0)
    assert(2 == runningAverage.getCount)
    assert((2.0 - runningAverage.getAverage).abs < epsilon)
  }


  test("testFullRunningAveragCopyConstructor") {
    val runningAverage: RunningAverage = new FullRunningAverage
    runningAverage.addDatum(1.0)
    runningAverage.addDatum(1.0)
    assert(2 == runningAverage.getCount)
    assert(1.0 - runningAverage.getAverage < epsilon)
    val copy: RunningAverage = new FullRunningAverage(runningAverage.getCount, runningAverage.getAverage)
    assert(2 == copy.getCount)
    assert(1.0 - copy.getAverage < epsilon)
  }



  // Inverted Running Average tests
  test("testInvertedRunningAverage") {
    val avg: RunningAverage = new FullRunningAverage
    val inverted: RunningAverage = new InvertedRunningAverage(avg)
    assert(0 == inverted.getCount)
    avg.addDatum(1.0)
    assert(1 == inverted.getCount)
    assert((1.0 + inverted.getAverage).abs < epsilon) // inverted.getAverage == -1.0
    avg.addDatum(2.0)
    assert(2 == inverted.getCount)
    assert((1.5 + inverted.getAverage).abs < epsilon) // inverted.getAverage == -1.5
  }

  test ("testInvertedRunningAverageAndStdDev") {
    val avg: RunningAverageAndStdDev = new FullRunningAverageAndStdDev
    val inverted: RunningAverageAndStdDev = new InvertedRunningAverageAndStdDev(avg)
    assert(0 == inverted.getCount)
    avg.addDatum(1.0)
    assert(1 == inverted.getCount)
    assert(((1.0 + inverted.getAverage).abs < epsilon)) // inverted.getAverage == -1.0
    avg.addDatum(2.0)
    assert(2 == inverted.getCount)
    assert((1.5 + inverted.getAverage).abs < epsilon) // inverted.getAverage == -1.5
    assert(((Math.sqrt(2.0) / 2.0) - inverted.getStandardDeviation).abs < epsilon)
  }


  // confusion Matrix tests
  val VALUES: Array[Array[Int]] = Array(Array(2, 3), Array(10, 20))
  val LABELS: Array[String] = Array("Label1", "Label2")
  val OTHER: Array[Int] = Array(3, 6)
  val DEFAULT_LABEL: String = "other"

  def fillConfusionMatrix(values: Array[Array[Int]], labels: Array[String], defaultLabel: String): ConfusionMatrix = {
    val labelList = Arrays.asList(labels(0),labels(1))
    val confusionMatrix: ConfusionMatrix = new ConfusionMatrix(labelList, defaultLabel)
    confusionMatrix.putCount("Label1", "Label1", values(0)(0))
    confusionMatrix.putCount("Label1", "Label2", values(0)(1))
    confusionMatrix.putCount("Label2", "Label1", values(1)(0))
    confusionMatrix.putCount("Label2", "Label2", values(1)(1))
    confusionMatrix.putCount("Label1", DEFAULT_LABEL, OTHER(0))
    confusionMatrix.putCount("Label2", DEFAULT_LABEL, OTHER(1))

    confusionMatrix
  }

  private def checkAccuracy(cm: ConfusionMatrix) {
    val labelstrs = cm.getLabels
    assert(3 == labelstrs.size)
    assert((25.0 - cm.getAccuracy("Label1")).abs < epsilon)
    assert((55.5555555 - cm.getAccuracy("Label2")).abs < epsilon)
    assert(true == Double.isNaN(cm.getAccuracy("other")))
  }

  private def checkValues(cm: ConfusionMatrix) {
    val counts: Array[Array[Int]] = cm.getConfusionMatrix
    cm.toString
    assert(counts.length == counts(0).length)
    assert(3 == counts.length)
    assert(VALUES(0)(0) == counts(0)(0))
    assert(VALUES(0)(1) == counts(0)(1))
    assert(VALUES(1)(0) == counts(1)(0))
    assert(VALUES(1)(1) == counts(1)(1))
    assert(true == Arrays.equals(new Array[Int](3), counts(2)))
    assert(OTHER(0) == counts(0)(2))
    assert(OTHER(1) == counts(1)(2))
    assert(3 == cm.getLabels.size)
    assert(true == cm.getLabels.contains(LABELS(0)))
    assert(true == cm.getLabels.contains(LABELS(1)))
    assert(true == cm.getLabels.contains(DEFAULT_LABEL))
  }

  test("testBuild"){
    val confusionMatrix: ConfusionMatrix = fillConfusionMatrix(VALUES, LABELS, DEFAULT_LABEL)
    checkValues(confusionMatrix)
    checkAccuracy(confusionMatrix)
  }

  test("GetMatrix") {
    val confusionMatrix: ConfusionMatrix = fillConfusionMatrix(VALUES, LABELS, DEFAULT_LABEL)
    val m: Matrix = confusionMatrix.getMatrix
    val rowLabels = m.getRowLabelBindings
    assert(confusionMatrix.getLabels.size == m.numCols)
    assert(true == rowLabels.keySet.contains(LABELS(0)))
    assert(true == rowLabels.keySet.contains(LABELS(1)))
    assert(true == rowLabels.keySet.contains(DEFAULT_LABEL))
    assert(2 == confusionMatrix.getCorrect(LABELS(0)))
    assert(20 == confusionMatrix.getCorrect(LABELS(1)))
    assert(0 == confusionMatrix.getCorrect(DEFAULT_LABEL))
  }

  /**
   * Example taken from
   * http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
   */
  test("testPrecisionRecallAndF1ScoreAsScikitLearn") {
    val labelList = Arrays.asList("0", "1", "2")
    val confusionMatrix: ConfusionMatrix = new ConfusionMatrix(labelList, "DEFAULT")
    confusionMatrix.putCount("0", "0", 2)
    confusionMatrix.putCount("1", "0", 1)
    confusionMatrix.putCount("1", "2", 1)
    confusionMatrix.putCount("2", "1", 2)
    val delta: Double = 0.001
    assert((0.222 - confusionMatrix.getWeightedPrecision).abs < delta)
    assert((0.333 - confusionMatrix.getWeightedRecall).abs < delta)
    assert((0.266 - confusionMatrix.getWeightedF1score).abs < delta)
  }



}
