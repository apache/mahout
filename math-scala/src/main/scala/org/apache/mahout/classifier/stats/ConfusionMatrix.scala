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
 */
class ConfusionMatrix(private var labels: util.Collection[String],
                      private var defaultLabel: String = "unknown")  {
  /**
   * Matrix Constructor
   * @param m
   */
   //  def this(m: Matrix) {
   //    this()
   //    confusionMatrix = new Array[Array[Int]](m.numRows, m.numRows)
   //    setMatrix(m)
   //  }

   // val LOG: Logger = LoggerFactory.getLogger(classOf[ConfusionMatrix])

  val confusionMatrix = Array.ofDim[Int](labels.size + 1, labels.size + 1)

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
        falsePositives += confusionMatrix(i)(labelId);
      }
    }

    if (truePositives + falsePositives == 0) {
      return 0
    } else {
      return (truePositives.asInstanceOf[Double]) / (truePositives + falsePositives)
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
      return 0
    } else {
      return (truePositives.asInstanceOf[Double]) / (truePositives + falseNegatives)
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
      return 0
    } else {
      return 2 * precision * recall / (precision + recall)
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
    for (i <- 0 until confusionMatrix.length) {
      a += confusionMatrix(i)(i)
      var br: Int = 0
      for (j <- 0 until confusionMatrix.length) {
        br += confusionMatrix(i)(j);
      }
      var bc: Int = 0
      //TODO: verify this as an iterator
      for (vec <- confusionMatrix) {
        bc += vec(i)
      }
      b += br * bc;
    }
    (samples * a - b) / (samples * samples - b)
  }

  def getCorrect(label: String): Int = {
    val labelId: Int = labelMap(label)
    return confusionMatrix(labelId)(labelId)
  }

  def getTotal(label: String): Int = {
    val labelId: Int = labelMap(label)
    var labelTotal: Int = 0
    for (i <- 0 until numLabels) {
      labelTotal += confusionMatrix(labelId)(i);
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
    for (d <- 0 until  confusionMatrix.length) {
      var total: Double = 0.0
      for (j <- 0 until  confusionMatrix.length) {
        total += confusionMatrix(d)(j)
      }
      summer.addDatum(confusionMatrix(d)(d) / (total + 0.000001));
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
        m.set(r, c, confusionMatrix(r)(c));
      }
    }

    for (entry <- labelMap.entrySet) {
      labels.put(entry.getKey, entry.getValue)
    }
    m.setRowLabelBindings(labels)
    m.setColumnLabelBindings(labels)
    return m
  }

  def setMatrix(m: Matrix) : Unit = {
    val length: Int = confusionMatrix.length
    if (m.numRows != m.numCols) {
      throw new IllegalArgumentException("ConfusionMatrix: matrix(" + m.numRows + ',' + m.numCols + ") must be square")
    }

    for (r <- 0 until length) {
      for (c <- 0 until length) {
        confusionMatrix(r)(c) = Math.round(m.get(r, c)).toInt;
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
        labelMap.put(sorted(i), i);
      }
    }
  }

  def verifyLabels(length: Int, sorted: Array[String]): Unit = {
    assert(sorted.length == length, "One label, one row")
    for (i <- 0 until length) {
      if (sorted(i) == null) {
        assert(false, "One label, one row");
      }
    }
  }

  def sortLabels(labels: java.util.Map[String, Integer]): Array[String] = {
    val sorted: Array[String] = new Array[String](labels.size)
    for (entry <- labels.entrySet) {
      sorted(entry.getValue) = entry.getKey
    }
    return sorted
  }
}
