/*
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

package org.apache.mahout.classifier.stats

import java.util
import org.apache.commons.math3.stat.descriptive.moment.Mean // This is brought in by mahout-math
import org.apache.mahout.math.{DenseMatrix, Matrix}
import scala.collection.mutable
import scala.collection.JavaConversions._

/**
 *
 * Ported from org.apache.mahout.classifier.ConfusionMatrix.java
 *
 * The ConfusionMatrix Class stores the result of Classification of a Test Dataset.
 *
 * The fact of whether there is a default is not stored. A row of zeros is the only indicator that there is no default.
 *
 * See http://en.wikipedia.org/wiki/Confusion_matrix for background
 *
 *
 * @param labels The labels to consider for classification
 * @param defaultLabel default unknown label
 */
class ConfusionMatrix(private var labels: util.Collection[String] = null,
                      private var defaultLabel: String = "unknown")  {
  /**
   * Matrix Constructor
   */
//   def this(m: Matrix) {
//     this()
//     confusionMatrix = Array.ofDim[Int](m.numRows, m.numRows)
//     setMatrix(m)
//   }

   // val LOG: Logger = LoggerFactory.getLogger(classOf[ConfusionMatrix])

  var confusionMatrix = Array.ofDim[Int](labels.size + 1, labels.size + 1)

  val labelMap = new mutable.HashMap[String,Integer]()

  var samples: Int = 0

  var i: Integer = 0
  for (label <- labels) {
    labelMap.put(label, i)
    i+=1
  }
  labelMap.put(defaultLabel, i)


  def getConfusionMatrix: Array[Array[Int]] = confusionMatrix

  def getLabels = labelMap.keys.toList

  def numLabels: Int = labelMap.size

  def getAccuracy(label: String): Double = {
    val labelId: Int = labelMap(label)
    var labelTotal: Int = 0
    var correct: Int = 0
    for (i <- 0 until numLabels) {
      labelTotal += confusionMatrix(labelId)(i)
      if (i == labelId) {
        correct += confusionMatrix(labelId)(i)
      }
    }

    100.0 * correct / labelTotal
  }

  def getAccuracy: Double = {
    var total: Int = 0
    var correct: Int = 0
    for (i <- 0 until numLabels) {
      for (j <- 0 until numLabels) {
        total += confusionMatrix(i)(j)
        if (i == j) {
          correct += confusionMatrix(i)(j)
        }
      }
    }

    100.0 * correct / total
  }

  /** Sum of true positives and false negatives */
  private def getActualNumberOfTestExamplesForClass(label: String): Int = {
    val labelId: Int = labelMap(label)
    var sum: Int = 0
    for (i <- 0 until numLabels) {
      sum += confusionMatrix(labelId)(i)
    }
    sum
  }

  def getPrecision(label: String): Double = {
    val labelId: Int = labelMap(label)
    val truePositives: Int = confusionMatrix(labelId)(labelId)
    var falsePositives: Int = 0

    for (i <- 0 until numLabels) {
      if (i != labelId) {
        falsePositives += confusionMatrix(i)(labelId)
      }
    }

    if (truePositives + falsePositives == 0) {
      0
    } else {
      truePositives.asInstanceOf[Double] / (truePositives + falsePositives)
    }
  }


  def getWeightedPrecision: Double = {
    val precisions: Array[Double] = new Array[Double](numLabels)
    val weights: Array[Double] = new Array[Double](numLabels)
    var index: Int = 0
    for (label <- labelMap.keys) {
      precisions(index) = getPrecision(label)
      weights(index) = getActualNumberOfTestExamplesForClass(label)
      index += 1
    }
    new Mean().evaluate(precisions, weights)
  }

  def getRecall(label: String): Double = {
    val labelId: Int = labelMap(label)
    val truePositives: Int = confusionMatrix(labelId)(labelId)
    var falseNegatives: Int = 0
    for (i <- 0 until numLabels) {
      if (i != labelId) {
        falseNegatives += confusionMatrix(labelId)(i)
      }
    }

    if (truePositives + falseNegatives == 0) {
      0
    } else {
      truePositives.asInstanceOf[Double] / (truePositives + falseNegatives)
    }
  }

  def getWeightedRecall: Double = {
    val recalls: Array[Double] = new Array[Double](numLabels)
    val weights: Array[Double] = new Array[Double](numLabels)
    var index: Int = 0
    for (label <- labelMap.keys) {
      recalls(index) = getRecall(label)
      weights(index) = getActualNumberOfTestExamplesForClass(label)
      index += 1
    }
    new Mean().evaluate(recalls, weights)
  }

  def getF1score(label: String): Double = {
    val precision: Double = getPrecision(label)
    val recall: Double = getRecall(label)
    if (precision + recall == 0) {
      0
    } else {
      2 * precision * recall / (precision + recall)
    }
  }

  def getWeightedF1score: Double = {
    val f1Scores: Array[Double] = new Array[Double](numLabels)
    val weights: Array[Double] = new Array[Double](numLabels)
    var index: Int = 0
    for (label <- labelMap.keys) {
      f1Scores(index) = getF1score(label)
      weights(index) = getActualNumberOfTestExamplesForClass(label)
      index += 1
    }
    new Mean().evaluate(f1Scores, weights)
  }

  def getReliability: Double = {
    var count: Int = 0
    var accuracy: Double = 0
    for (label <- labelMap.keys) {
      if (!(label == defaultLabel)) {
        accuracy += getAccuracy(label)
      }
      count += 1
    }
    accuracy / count
  }

  /**
   * Accuracy v.s. randomly classifying all samples.
   * kappa() = (totalAccuracy() - randomAccuracy()) / (1 - randomAccuracy())
   * Cohen, Jacob. 1960. A coefficient of agreement for nominal scales.
   * Educational And Psychological Measurement 20:37-46.
   *
   * Formula and variable names from:
   * http://www.yale.edu/ceo/OEFS/Accuracy.pdf
   *
   * @return double
   */
  def getKappa: Double = {
    var a: Double = 0.0
    var b: Double = 0.0
    for (i <- confusionMatrix.indices) {
      a += confusionMatrix(i)(i)
      var br: Int = 0
      for (j <- confusionMatrix.indices) {
        br += confusionMatrix(i)(j)
      }
      var bc: Int = 0
      //TODO: verify this as an iterator
      for (vec <- confusionMatrix) {
        bc += vec(i)
      }
      b += br * bc
    }
    (samples * a - b) / (samples * samples - b)
  }

  def getCorrect(label: String): Int = {
    val labelId: Int = labelMap(label)
    confusionMatrix(labelId)(labelId)
  }

  def getTotal(label: String): Int = {
    val labelId: Int = labelMap(label)
    var labelTotal: Int = 0
    for (i <- 0 until numLabels) {
      labelTotal += confusionMatrix(labelId)(i)
    }
    labelTotal
  }

  /**
   * Standard deviation of normalized producer accuracy
   * Not a standard score
   * @return double
   */
  def getNormalizedStats: RunningAverageAndStdDev = {
    val summer = new FullRunningAverageAndStdDev()
    for (d <- confusionMatrix.indices) {
      var total: Double = 0.0
      for (j <- confusionMatrix.indices) {
        total += confusionMatrix(d)(j)
      }
      summer.addDatum(confusionMatrix(d)(d) / (total + 0.000001))
    }
    summer
  }

  def addInstance(correctLabel: String, classifiedResult: ClassifierResult): Unit = {
    samples += 1
    incrementCount(correctLabel, classifiedResult.getLabel)
  }

  def addInstance(correctLabel: String, classifiedLabel: String): Unit = {
    samples += 1
    incrementCount(correctLabel, classifiedLabel)
  }

  def getCount(correctLabel: String, classifiedLabel: String): Int = {
    if (!labelMap.containsKey(correctLabel)) {
    //  LOG.warn("Label {} did not appear in the training examples", correctLabel)
      return 0
    }
    assert(labelMap.containsKey(classifiedLabel), "Label not found: " + classifiedLabel)
    val correctId: Int = labelMap(correctLabel)
    val classifiedId: Int = labelMap(classifiedLabel)
    confusionMatrix(correctId)(classifiedId)
  }

  def putCount(correctLabel: String, classifiedLabel: String, count: Int): Unit = {
    if (!labelMap.containsKey(correctLabel)) {
    //  LOG.warn("Label {} did not appear in the training examples", correctLabel)
      return
    }
    assert(labelMap.containsKey(classifiedLabel), "Label not found: " + classifiedLabel)
    val correctId: Int = labelMap(correctLabel)
    val classifiedId: Int = labelMap(classifiedLabel)
    if (confusionMatrix(correctId)(classifiedId) == 0.0 && count != 0) {
      samples += 1
    }
    confusionMatrix(correctId)(classifiedId) = count
  }

  def incrementCount(correctLabel: String, classifiedLabel: String, count: Int): Unit = {
    putCount(correctLabel, classifiedLabel, count + getCount(correctLabel, classifiedLabel))
  }

  def incrementCount(correctLabel: String, classifiedLabel: String): Unit = {
    incrementCount(correctLabel, classifiedLabel, 1)
  }

  def getDefaultLabel: String = {
    defaultLabel
  }

  def merge(b: ConfusionMatrix): ConfusionMatrix = {
    assert(labelMap.size == b.getLabels.size, "The label sizes do not match")
    for (correctLabel <- this.labelMap.keys) {
      for (classifiedLabel <- this.labelMap.keys) {
        incrementCount(correctLabel, classifiedLabel, b.getCount(correctLabel, classifiedLabel))
      }
    }
    this
  }

  def getMatrix: Matrix = {
    val length: Int = confusionMatrix.length
    val m: Matrix = new DenseMatrix(length, length)

    val labels: java.util.HashMap[String, Integer] = new java.util.HashMap()

    for (r <- 0 until length) {
      for (c <- 0 until length) {
        m.set(r, c, confusionMatrix(r)(c))
      }
    }

    for (entry <- labelMap.entrySet) {
      labels.put(entry.getKey, entry.getValue)
    }
    m.setRowLabelBindings(labels)
    m.setColumnLabelBindings(labels)

    m
  }

  def setMatrix(m: Matrix) : Unit = {
    val length: Int = confusionMatrix.length
    if (m.numRows != m.numCols) {
      throw new IllegalArgumentException("ConfusionMatrix: matrix(" + m.numRows + ',' + m.numCols + ") must be square")
    }

    for (r <- 0 until length) {
      for (c <- 0 until length) {
        confusionMatrix(r)(c) = Math.round(m.get(r, c)).toInt
      }
    }

    var labels = m.getRowLabelBindings
    if (labels == null) {
      labels = m.getColumnLabelBindings
    }

    if (labels != null) {
      val sorted: Array[String] = sortLabels(labels)
      verifyLabels(length, sorted)
      labelMap.clear
      for (i <- 0 until length) {
        labelMap.put(sorted(i), i)
      }
    }
  }

  def verifyLabels(length: Int, sorted: Array[String]): Unit = {
    assert(sorted.length == length, "One label, one row")
    for (i <- 0 until length) {
      if (sorted(i) == null) {
        assert(assertion = false, "One label, one row")
      }
    }
  }

  def sortLabels(labels: java.util.Map[String, Integer]): Array[String] = {
    val sorted: Array[String] = new Array[String](labels.size)
    for (entry <- labels.entrySet) {
      sorted(entry.getValue) = entry.getKey
    }

    sorted
  }

  /**
   * This is overloaded. toString() is not a formatted report you print for a manager :)
   * Assume that if there are no default assignments, the default feature was not used
   */
  override def toString: String = {

    val returnString: StringBuilder = new StringBuilder(200)

    returnString.append("=======================================================").append('\n')
    returnString.append("Confusion Matrix\n")
    returnString.append("-------------------------------------------------------").append('\n')

    val unclassified: Int = getTotal(defaultLabel)

    for (entry <- this.labelMap.entrySet) {
      if (!((entry.getKey == defaultLabel) && unclassified == 0)) {
        returnString.append(getSmallLabel(entry.getValue) + "     ").append('\t')
      }
    }

    returnString.append("<--Classified as").append('\n')

    for (entry <- this.labelMap.entrySet) {
      if (!((entry.getKey == defaultLabel) && unclassified == 0)) {
        val correctLabel: String = entry.getKey
        var labelTotal: Int = 0

        for (classifiedLabel <- this.labelMap.keySet) {
          if (!((classifiedLabel == defaultLabel) && unclassified == 0)) {
            returnString.append(Integer.toString(getCount(correctLabel, classifiedLabel)) + "     ")
                        .append('\t')
            labelTotal += getCount(correctLabel, classifiedLabel)
          }
        }
        returnString.append(" |  ").append(String.valueOf(labelTotal) + "      ")
                    .append('\t')
                    .append(getSmallLabel(entry.getValue) + "     ")
                    .append(" = ")
                    .append(correctLabel)
                    .append('\n')
      }
    }

    if (unclassified > 0) {
      returnString.append("Default Category: ")
                  .append(defaultLabel)
                  .append(": ")
                  .append(unclassified)
                  .append('\n')
    }
    returnString.append('\n')

    returnString.toString()
  }


  def getSmallLabel(i: Int): String = {
    var value: Int = i
    val returnString: StringBuilder = new StringBuilder
    do {
      val n: Int = value % 26
      returnString.insert(0, ('a' + n).asInstanceOf[Char])
      value /= 26
    } while (value > 0)

    returnString.toString()
  }


}
