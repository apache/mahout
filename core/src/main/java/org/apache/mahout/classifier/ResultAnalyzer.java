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

import org.apache.commons.lang.StringUtils;
import org.apache.mahout.math.stats.OnlineSummarizer;

/**
 * ResultAnalyzer captures the classification statistics and displays in a tabular manner
 */
public class ResultAnalyzer {
  
  private final ConfusionMatrix confusionMatrix;
  private final OnlineSummarizer summarizer;
  private boolean hasLL = false;
  
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
    if (hasLL) {
      returnString.append("\n\n");
      returnString.append("Avg. Log-likelihood: ").append(summarizer.getMean()).append(" 25%-ile: ").append(summarizer.getQuartile(1))
              .append(" 75%-ile: ").append(summarizer.getQuartile(2));
    }

    return returnString.toString();
  }
}
