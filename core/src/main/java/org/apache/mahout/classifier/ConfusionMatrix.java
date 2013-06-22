/**
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * Licensed to the Apache Software Foundation (ASF) under one or more
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

import java.util.Collection;
import java.util.Collections;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverageAndStdDev;
import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;

/**
 * The ConfusionMatrix Class stores the result of Classification of a Test Dataset.
 * 
 * The fact of whether there is a default is not stored. A row of zeros is the only indicator that there is no default.
 * 
 * See http://en.wikipedia.org/wiki/Confusion_matrix for background
 */
public class ConfusionMatrix {
  private final Map<String,Integer> labelMap = Maps.newLinkedHashMap();
  private final int[][] confusionMatrix;
  private int samples = 0;
  private String defaultLabel = "unknown";
  
  public ConfusionMatrix(Collection<String> labels, String defaultLabel) {
    confusionMatrix = new int[labels.size() + 1][labels.size() + 1];
    this.defaultLabel = defaultLabel;
    int i = 0;
    for (String label : labels) {
      labelMap.put(label, i++);
    }
    labelMap.put(defaultLabel, i);
  }
  
  public ConfusionMatrix(Matrix m) {
    confusionMatrix = new int[m.numRows()][m.numRows()];
    setMatrix(m);
  }
  
  public int[][] getConfusionMatrix() {
    return confusionMatrix;
  }
  
  public Collection<String> getLabels() {
    return Collections.unmodifiableCollection(labelMap.keySet());
  }
  
  public double getAccuracy(String label) {
    int labelId = labelMap.get(label);
    int labelTotal = 0;
    int correct = 0;
    for (int i = 0; i < labelMap.size(); i++) {
      labelTotal += confusionMatrix[labelId][i];
      if (i == labelId) {
        correct += confusionMatrix[labelId][i];
      }
    }
    return 100.0 * correct / labelTotal;
  }

  // Producer accuracy
  public double getAccuracy() {
    int total = 0;
    int correct = 0;
    for (int i = 0; i < labelMap.size(); i++) {
      for (int j = 0; j < labelMap.size(); j++) {
        total += confusionMatrix[i][j];
        if (i == j) {
          correct += confusionMatrix[i][j];
        }
      }
    }
    return 100.0 * correct / total;
  }
  
  // User accuracy 
  public double getReliability() {
    int count = 0;
    double accuracy = 0;
    for (String label: labelMap.keySet()) {
      if (!label.equals(defaultLabel)) {
        accuracy += getAccuracy(label);
      }
      count++;
    }
    return accuracy / count;
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
  public double getKappa() {
    double a = 0.0;
    double b = 0.0;
    for (int i = 0; i < confusionMatrix.length; i++) {
      a += confusionMatrix[i][i];
      double br = 0;
      for (int j = 0; j < confusionMatrix.length; j++) {
        br += confusionMatrix[i][j];
      }
      double bc = 0;
      for (int[] vec : confusionMatrix) {
        bc += vec[i];
      }
      b += br * bc;
    }
    return (samples * a - b) / (samples * samples - b);
  }
  
  /**
   * Standard deviation of normalized producer accuracy
   * Not a standard score
   * @return double
   */
  public RunningAverageAndStdDev getNormalizedStats() {
    RunningAverageAndStdDev summer = new FullRunningAverageAndStdDev();
    for (int d = 0; d < confusionMatrix.length; d++) {
      double total = 0;
      for (int j = 0; j < confusionMatrix.length; j++) {
        total += confusionMatrix[d][j];
      }
      summer.addDatum(confusionMatrix[d][d] / (total + 0.000001));
    }
    
    return summer;
  }
   
  public int getCorrect(String label) {
    int labelId = labelMap.get(label);
    return confusionMatrix[labelId][labelId];
  }
  
  public int getTotal(String label) {
    int labelId = labelMap.get(label);
    int labelTotal = 0;
    for (int i = 0; i < labelMap.size(); i++) {
      labelTotal += confusionMatrix[labelId][i];
    }
    return labelTotal;
  }
  
  public void addInstance(String correctLabel, ClassifierResult classifiedResult) {
    samples++;
    incrementCount(correctLabel, classifiedResult.getLabel());
  }
  
  public void addInstance(String correctLabel, String classifiedLabel) {
    samples++;
    incrementCount(correctLabel, classifiedLabel);
  }
  
  public int getCount(String correctLabel, String classifiedLabel) {
    Preconditions.checkArgument(labelMap.containsKey(correctLabel), "Label not found: " + correctLabel);
    Preconditions.checkArgument(labelMap.containsKey(classifiedLabel), "Label not found: " + classifiedLabel);
    int correctId = labelMap.get(correctLabel);
    int classifiedId = labelMap.get(classifiedLabel);
    return confusionMatrix[correctId][classifiedId];
  }
  
  public void putCount(String correctLabel, String classifiedLabel, int count) {
    Preconditions.checkArgument(labelMap.containsKey(correctLabel), "Label not found: " + correctLabel);
    Preconditions.checkArgument(labelMap.containsKey(classifiedLabel), "Label not found: " + classifiedLabel);
    int correctId = labelMap.get(correctLabel);
    int classifiedId = labelMap.get(classifiedLabel);
    if (confusionMatrix[correctId][classifiedId] == 0.0 && count != 0) {
      samples++;
    }
    confusionMatrix[correctId][classifiedId] = count;
  }
  
  public String getDefaultLabel() {
    return defaultLabel;
  }
  
  public void incrementCount(String correctLabel, String classifiedLabel, int count) {
    putCount(correctLabel, classifiedLabel, count + getCount(correctLabel, classifiedLabel));
  }
  
  public void incrementCount(String correctLabel, String classifiedLabel) {
    incrementCount(correctLabel, classifiedLabel, 1);
  }
  
  public ConfusionMatrix merge(ConfusionMatrix b) {
    Preconditions.checkArgument(labelMap.size() == b.getLabels().size(), "The label sizes do not match");
    for (String correctLabel : this.labelMap.keySet()) {
      for (String classifiedLabel : this.labelMap.keySet()) {
        incrementCount(correctLabel, classifiedLabel, b.getCount(correctLabel, classifiedLabel));
      }
    }
    return this;
  }
  
  public Matrix getMatrix() {
    int length = confusionMatrix.length;
    Matrix m = new DenseMatrix(length, length);
    for (int r = 0; r < length; r++) {
      for (int c = 0; c < length; c++) {
        m.set(r, c, confusionMatrix[r][c]);
      }
    }
    Map<String,Integer> labels = Maps.newHashMap();
    for (Map.Entry<String, Integer> entry : labelMap.entrySet()) {
      labels.put(entry.getKey(), entry.getValue());
    }
    m.setRowLabelBindings(labels);
    m.setColumnLabelBindings(labels);
    return m;
  }
  
  public void setMatrix(Matrix m) {
    int length = confusionMatrix.length;
    if (m.numRows() != m.numCols()) {
      throw new IllegalArgumentException(
          "ConfusionMatrix: matrix(" + m.numRows() + ',' + m.numCols() + ") must be square");
    }
    for (int r = 0; r < length; r++) {
      for (int c = 0; c < length; c++) {
        confusionMatrix[r][c] = (int) Math.round(m.get(r, c));
      }
    }
    Map<String,Integer> labels = m.getRowLabelBindings();
    if (labels == null) {
      labels = m.getColumnLabelBindings();
    }
    if (labels != null) {
      String[] sorted = sortLabels(labels);
      verifyLabels(length, sorted);
      labelMap.clear();
      for (int i = 0; i < length; i++) {
        labelMap.put(sorted[i], i);
      }
    }
  }
  
  private static String[] sortLabels(Map<String,Integer> labels) {
    String[] sorted = new String[labels.size()];
    for (Map.Entry<String,Integer> entry : labels.entrySet()) {
      sorted[entry.getValue()] = entry.getKey();
    }
    return sorted;
  }
  
  private static void verifyLabels(int length, String[] sorted) {
    Preconditions.checkArgument(sorted.length == length, "One label, one row");
    for (int i = 0; i < length; i++) {
      if (sorted[i] == null) {
        Preconditions.checkArgument(false, "One label, one row");
      }
    }
  }
  
  /**
   * This is overloaded. toString() is not a formatted report you print for a manager :)
   * Assume that if there are no default assignments, the default feature was not used
   */
  @Override
  public String toString() {
    StringBuilder returnString = new StringBuilder(200);
    returnString.append("=======================================================").append('\n');
    returnString.append("Confusion Matrix\n");
    returnString.append("-------------------------------------------------------").append('\n');
    
    int unclassified = getTotal(defaultLabel);
    for (Map.Entry<String,Integer> entry : this.labelMap.entrySet()) {
      if (entry.getKey().equals(defaultLabel) && unclassified == 0) {
        continue;
      }
      
      returnString.append(StringUtils.rightPad(getSmallLabel(entry.getValue()), 5)).append('\t');
    }
    
    returnString.append("<--Classified as").append('\n');
    for (Map.Entry<String,Integer> entry : this.labelMap.entrySet()) {
      if (entry.getKey().equals(defaultLabel) && unclassified == 0) {
        continue;
      }
      String correctLabel = entry.getKey();
      int labelTotal = 0;
      for (String classifiedLabel : this.labelMap.keySet()) {
        if (classifiedLabel.equals(defaultLabel) && unclassified == 0) {
          continue;
        }
        returnString.append(
            StringUtils.rightPad(Integer.toString(getCount(correctLabel, classifiedLabel)), 5)).append('\t');
        labelTotal += getCount(correctLabel, classifiedLabel);
      }
      returnString.append(" |  ").append(StringUtils.rightPad(String.valueOf(labelTotal), 6)).append('\t')
      .append(StringUtils.rightPad(getSmallLabel(entry.getValue()), 5))
      .append(" = ").append(correctLabel).append('\n');
    }
    if (unclassified > 0) {
      returnString.append("Default Category: ").append(defaultLabel).append(": ").append(unclassified).append('\n');
    }
    returnString.append('\n');
    return returnString.toString();
  }
  
  static String getSmallLabel(int i) {
    int val = i;
    StringBuilder returnString = new StringBuilder();
    do {
      int n = val % 26;
      returnString.insert(0, (char) ('a' + n));
      val /= 26;
    } while (val > 0);
    return returnString.toString();
  }

}
