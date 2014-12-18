package org.apache.mahout.classifier.stats

import java.text.{DecimalFormat, NumberFormat}
import java.util
//import org.apache.commons.lang3.StringUtils
import org.apache.mahout.math.stats.OnlineSummarizer


/**
 * Result of a document classification. The label and the associated score (usually probabilty)
 */
class ClassifierResult (private var label: String = null,
                        private var score: Double = 0.0,
                        private var logLikelihood: Double = Integer.MAX_VALUE.toDouble) {

  def getLogLikelihood: Double = {
    return logLikelihood
  }

  def setLogLikelihood(logLikelihood: Double) {
    this.logLikelihood = logLikelihood
  }

  def getLabel: String = {
     label
  }

  def getScore: Double = {
    return score
  }

  def setLabel(label: String) {
    this.label = label
  }

  def setScore(score: Double) {
    this.score = score
  }

  override def toString: String = {
    return "ClassifierResult{" + "category='" + label + '\'' + ", score=" + score + '}'
  }

}

/** ResultAnalyzer captures the classification statistics and displays in a tabular manner */
class ResultAnalyzer(private val labelSet: util.Collection[String], defaultLabel: String) {

  val confusionMatrix = new ConfusionMatrix(labelSet, defaultLabel)
  val summarizer = new OnlineSummarizer

  private var hasLL: Boolean = false
  private var correctlyClassified: Int = 0
  private var incorrectlyClassified: Int = 0


  def getConfusionMatrix: ConfusionMatrix = {
    return this.confusionMatrix
  }

  /**
   *
   * @param correctLabel
   * The correct label
   * @param classifiedResult
   * The classified result
   * @return whether the instance was correct or not
   */
  def addInstance(correctLabel: String, classifiedResult: ClassifierResult): Boolean = {
    val result: Boolean = correctLabel == classifiedResult.getLabel
    if (result) {
      correctlyClassified += 1
    }
    else {
      incorrectlyClassified += 1
    }
    confusionMatrix.addInstance(correctLabel, classifiedResult)
    if (classifiedResult.getLogLikelihood != Integer.MAX_VALUE.toDouble) {
      summarizer.add(classifiedResult.getLogLikelihood)
      hasLL = true
    }
    return result
  }

  override def toString: String = {
    val returnString: StringBuilder = new StringBuilder
    returnString.append('\n')
    returnString.append("=======================================================\n")
    returnString.append("Summary\n")
    returnString.append("-------------------------------------------------------\n")
    val totalClassified: Int = correctlyClassified + incorrectlyClassified
    val percentageCorrect: Double = 100.asInstanceOf[Double] * correctlyClassified / totalClassified
    val percentageIncorrect: Double = 100.asInstanceOf[Double] * incorrectlyClassified / totalClassified
    val decimalFormatter: NumberFormat = new DecimalFormat("0.####")
    returnString.append("Correctly Classified Instances")
                .append(": ")
                .append(Integer.toString(correctlyClassified))
                .append('\t')
                .append(decimalFormatter.format(percentageCorrect))
                .append("%\n")
    returnString.append("Incorrectly Classified Instances")
                .append(": ")
                .append(Integer.toString(incorrectlyClassified))
                .append('\t')
                .append(decimalFormatter.format(percentageIncorrect))
                .append("%\n")
    returnString.append("Total Classified Instances")
                .append(": ")
                .append(Integer.toString(totalClassified))
                .append('\n')
    returnString.append('\n')
    returnString.append(confusionMatrix)
    returnString.append("=======================================================\n")
    returnString.append("Statistics\n")
    returnString.append("-------------------------------------------------------\n")
    val normStats: RunningAverageAndStdDev = confusionMatrix.getNormalizedStats
    returnString.append("Kappa")
                .append(decimalFormatter.format(confusionMatrix.getKappa))
                .append('\n')
    returnString.append("Accuracy")
                .append(decimalFormatter.format(confusionMatrix.getAccuracy))
                .append("%\n")
    returnString.append("Reliability")
                .append(decimalFormatter.format(normStats.getAverage * 100.00000001))
                .append("%\n")
    returnString.append("Reliability (standard deviation)")
                .append(decimalFormatter.format(normStats.getStandardDeviation))
                .append('\n')
    returnString.append("Weighted precision")
                .append(decimalFormatter.format(confusionMatrix.getWeightedPrecision))
                .append('\n')
    returnString.append("Weighted recall")
                .append(decimalFormatter.format(confusionMatrix.getWeightedRecall))
                .append('\n')
    returnString.append("Weighted F1 score")
                .append(decimalFormatter.format(confusionMatrix.getWeightedF1score))
                .append('\n')
    if (hasLL) {
      returnString.append("Log-likelihood")
                  .append("mean      : ")
                  .append(decimalFormatter.format(summarizer.getMean))
                  .append('\n')
      returnString.append("25%-ile   : ")
                  .append(decimalFormatter.format(summarizer.getQuartile(1)))
                  .append('\n')
      returnString.append("75%-ile   : ")
                  .append(decimalFormatter.format(summarizer.getQuartile(3)))
                  .append('\n')
    }
    return returnString.toString
  }


}

/**
 *
 * Interface for classes that can keep track of a running average of a series of numbers. One can add to or
 * remove from the series, as well as update a datum in the series. The class does not actually keep track of
 * the series of values, just its running average, so it doesn't even matter if you remove/change a value that
 * wasn't added.
 *
 * Ported from org.apache.mahout.cf.taste.impl.common.RunningAverage.java
 */
abstract trait RunningAverage {
  /**
   * @param datum
   * new item to add to the running average
   * @throws IllegalArgumentException
   * if datum is { @link Double#NaN}
   */
  def addDatum(datum: Double)

  /**
   * @param datum
   * item to remove to the running average
   * @throws IllegalArgumentException
   * if datum is { @link Double#NaN}
   * @throws IllegalStateException
   * if count is 0
   */
  def removeDatum(datum: Double)

  /**
   * @param delta
   * amount by which to change a datum in the running average
   * @throws IllegalArgumentException
   * if delta is { @link Double#NaN}
   * @throws IllegalStateException
   * if count is 0
   */
  def changeDatum(delta: Double)

  def getCount: Int

  def getAverage: Double

  /**
   * @return a (possibly immutable) object whose average is the negative of this object's
   */
  def inverse: RunningAverage
}

/**
 *
 * Extends {@link RunningAverage} by adding standard deviation too.
 *
 * Ported from org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev.java
 */
abstract trait RunningAverageAndStdDev extends RunningAverage {
  /** @return standard deviation of data */
  def getStandardDeviation: Double

  /**
   * @return a (possibly immutable) object whose average is the negative of this object's
   */
  def inverse: RunningAverageAndStdDev
}


class InvertedRunningAverage(private val delegate: RunningAverage) extends RunningAverage {

  def addDatum(datum: Double) {
    throw new UnsupportedOperationException
  }

  def removeDatum(datum: Double) {
    throw new UnsupportedOperationException
  }

  def changeDatum(delta: Double) {
    throw new UnsupportedOperationException
  }

  def getCount: Int = {
     delegate.getCount
  }

  def getAverage: Double = {
     -delegate.getAverage
  }

  def inverse: RunningAverage = {
     delegate
  }

}

/**
 *
 * A simple class that can keep track of a running average of a series of numbers. One can add to or remove
 * from the series, as well as update a datum in the series. The class does not actually keep track of the
 * series of values, just its running average, so it doesn't even matter if you remove/change a value that
 * wasn't added.
 *
 * Ported from org.apache.mahout.cf.taste.impl.common.FullRunningAverage.java
 */
class FullRunningAverage(private var count: Int = 0, private var average: Double = Double.NaN ) extends RunningAverage {

  /**
   * @param datum
   * new item to add to the running average
   */
  def addDatum(datum: Double) {
    count += 1
    if (count == 1) {
      average = datum
    }
    else {
      average = average * (count - 1) / count + datum / count
    }
  }

  /**
   * @param datum
   * item to remove to the running average
   * @throws IllegalStateException
   * if count is 0
   */
  def removeDatum(datum: Double) {
    if (count == 0) {
      throw new IllegalStateException
    }
    count -= 1
    if (count == 0) {
      average = Double.NaN
    }
    else {
      average = average * (count + 1) / count - datum / count
    }
  }

  /**
   * @param delta
   * amount by which to change a datum in the running average
   * @throws IllegalStateException
   * if count is 0
   */
  def changeDatum(delta: Double) {
    if (count == 0) {
      throw new IllegalStateException
    }
    average += delta / count
  }

  def getCount: Int = {
    count
  }

  def getAverage: Double = {
    average
  }

  def inverse: RunningAverage = {
    new InvertedRunningAverage(this)
  }

  override def toString: String = {
    String.valueOf(average)
  }


}


/**
 *
 * Extends {@link FullRunningAverage} to add a running standard deviation computation.
 * Uses Welford's method, as described at http://www.johndcook.com/standard_deviation.html
 *
 * Ported from org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev.java
 */
class FullRunningAverageAndStdDev(private var count: Int = 0,
                                        private var average: Double = 0.0,
                                        private var mk: Double = 0.0,
                                        private var sk: Double = 0.0) extends FullRunningAverage with RunningAverageAndStdDev {

  var stdDev: Double = 0.0

  recomputeStdDev

  def getMk: Double = {
     mk
  }

  def getSk: Double = {
    sk
  }

  def getStandardDeviation: Double = {
    stdDev
  }

  override def addDatum(datum: Double) {
    super.addDatum(datum)
    val count: Int = getCount
    if (count == 1) {
      mk = datum
      sk = 0.0
    }
    else {
      val oldmk: Double = mk
      val diff: Double = datum - oldmk
      mk += diff / count
      sk += diff * (datum - mk)
    }
    recomputeStdDev
  }

  override def removeDatum(datum: Double) {
    val oldCount: Int = getCount
    super.removeDatum(datum)
    val oldmk: Double = mk
    mk = (oldCount * oldmk - datum) / (oldCount - 1)
    sk -= (datum - mk) * (datum - oldmk)
    recomputeStdDev
  }

  /**
   * @throws UnsupportedOperationException
   */
  override def changeDatum(delta: Double) {
    throw new UnsupportedOperationException
  }

  private def recomputeStdDev {
    val count: Int = getCount
    stdDev = if (count > 1) Math.sqrt(sk / (count - 1)) else Double.NaN
  }

  override def inverse: RunningAverageAndStdDev = {
     new InvertedRunningAverageAndStdDev(this)
  }

  override def toString: String = {
     String.valueOf(String.valueOf(getAverage) + ',' + stdDev)
  }

}


/**
 *
 * @param delegate RunningAverageAndStdDev instance
 *
 * Ported from org.apache.mahout.cf.taste.impl.common.InvertedRunningAverageAndStdDev.java
 */
class InvertedRunningAverageAndStdDev(private val delegate: RunningAverageAndStdDev) extends RunningAverageAndStdDev {

  def addDatum(datum: Double) {
    throw new UnsupportedOperationException
  }

  def removeDatum(datum: Double) {
    throw new UnsupportedOperationException
  }

  def changeDatum(delta: Double) {
    throw new UnsupportedOperationException
  }

  def getCount: Int = {
     delegate.getCount
  }

  def getAverage: Double = {
     -delegate.getAverage
  }

  def getStandardDeviation: Double = {
     delegate.getStandardDeviation
  }

  def inverse: RunningAverageAndStdDev = {
     delegate
  }

}




