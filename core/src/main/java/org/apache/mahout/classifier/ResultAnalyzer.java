/**
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

package org.apache.mahout.classifier;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Collection;

import org.apache.commons.lang3.StringUtils;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.math.stats.OnlineSummarizer;

/**
 * ResultAnalyzer captures the classification statistics and displays in a tabular manner
 */
public class ResultAnalyzer {
  
  private final ConfusionMatrix confusionMatrix;
  private final OnlineSummarizer summarizer;
  private boolean hasLL;
  /*
   * === Summary ===
   * 
   * Correctly Classified Instances 635 92.9722 % Incorrectly Classified Instances 48 7.0278 % Kappa statistic
   * 0.923 Mean absolute error 0.0096 Root mean squared error 0.0817 Relative absolute error 9.9344 % Root
   * relative squared error 37.2742 % Total Number of Instances 683
   */
  private int correctlyClassified;
  private int incorrectlyClassified;
  
  public ResultAnalyzer(Collection<String> labelSet, String defaultLabel) {
    confusionMatrix = new ConfusionMatrix(labelSet, defaultLabel);
    summarizer = new OnlineSummarizer();
  }
  
  public ConfusionMatrix getConfusionMatrix() {
    return this.confusionMatrix;
  }
  
  /**
   * 
   * @param correctLabel
   *          The correct label
   * @param classifiedResult
   *          The classified result
   * @return whether the instance was correct or not
   */
  public boolean addInstance(String correctLabel, ClassifierResult classifiedResult) {
    boolean result = correctLabel.equals(classifiedResult.getLabel());
    if (result) {
      correctlyClassified++;
    } else {
      incorrectlyClassified++;
    }
    confusionMatrix.addInstance(correctLabel, classifiedResult);
    if (classifiedResult.getLogLikelihood() != Double.MAX_VALUE) {
      summarizer.add(classifiedResult.getLogLikelihood());
      hasLL = true;
    }
    return result;
  }
  
  @Override
  public String toString() {
    StringBuilder returnString = new StringBuilder();
   
    returnString.append('\n'); 
    returnString.append("=======================================================\n");
    returnString.append("Summary\n");
    returnString.append("-------------------------------------------------------\n");
    int totalClassified = correctlyClassified + incorrectlyClassified;
    double percentageCorrect = (double) 100 * correctlyClassified / totalClassified;
    double percentageIncorrect = (double) 100 * incorrectlyClassified / totalClassified;
    NumberFormat decimalFormatter = new DecimalFormat("0.####");
    
    returnString.append(StringUtils.rightPad("Correctly Classified Instances", 40)).append(": ").append(
      StringUtils.leftPad(Integer.toString(correctlyClassified), 10)).append('\t').append(
      StringUtils.leftPad(decimalFormatter.format(percentageCorrect), 10)).append("%\n");
    returnString.append(StringUtils.rightPad("Incorrectly Classified Instances", 40)).append(": ").append(
      StringUtils.leftPad(Integer.toString(incorrectlyClassified), 10)).append('\t').append(
      StringUtils.leftPad(decimalFormatter.format(percentageIncorrect), 10)).append("%\n");
    returnString.append(StringUtils.rightPad("Total Classified Instances", 40)).append(": ").append(
      StringUtils.leftPad(Integer.toString(totalClassified), 10)).append('\n');
    returnString.append('\n');
    
    returnString.append(confusionMatrix);
    returnString.append("=======================================================\n");
    returnString.append("Statistics\n");
    returnString.append("-------------------------------------------------------\n");
    
    RunningAverageAndStdDev normStats = confusionMatrix.getNormalizedStats();
    returnString.append(StringUtils.rightPad("Kappa", 40)).append(
      StringUtils.leftPad(decimalFormatter.format(confusionMatrix.getKappa()), 10)).append('\n');
    returnString.append(StringUtils.rightPad("Accuracy", 40)).append(
      StringUtils.leftPad(decimalFormatter.format(confusionMatrix.getAccuracy()), 10)).append("%\n");
    returnString.append(StringUtils.rightPad("Reliability", 40)).append(
      StringUtils.leftPad(decimalFormatter.format(normStats.getAverage() * 100.00000001), 10)).append("%\n");
    returnString.append(StringUtils.rightPad("Reliability (standard deviation)", 40)).append(
      StringUtils.leftPad(decimalFormatter.format(normStats.getStandardDeviation()), 10)).append('\n'); 
    
    if (hasLL) {
      returnString.append(StringUtils.rightPad("Log-likelihood", 30)).append("mean      : ").append(
        StringUtils.leftPad(decimalFormatter.format(summarizer.getMean()), 10)).append('\n');
      returnString.append(StringUtils.rightPad("", 30)).append(StringUtils.rightPad("25%-ile   : ", 10)).append(
        StringUtils.leftPad(decimalFormatter.format(summarizer.getQuartile(1)), 10)).append('\n');
      returnString.append(StringUtils.rightPad("", 30)).append(StringUtils.rightPad("75%-ile   : ", 10)).append(
        StringUtils.leftPad(decimalFormatter.format(summarizer.getQuartile(3)), 10)).append('\n');
    }

    return returnString.toString();
  }
}
