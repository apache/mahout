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
import java.util.LinkedHashMap;
import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.commons.lang.StringUtils;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import com.google.common.base.Preconditions;

/**
 * The ConfusionMatrix Class stores the result of Classification of a Test Dataset.
 * 
 * See http://en.wikipedia.org/wiki/Confusion_matrix for background
 */
public class ConfusionMatrix {

  private final Map<String,Integer> labelMap = new LinkedHashMap<String,Integer>();
  private final int[][] confusionMatrix;
  private String defaultLabel = "unknown";
  
  public ConfusionMatrix(Collection<String> labels, String defaultLabel) {
    confusionMatrix = new int[labels.size() + 1][labels.size() + 1];
    this.defaultLabel = defaultLabel;
    for (String label : labels) {
      labelMap.put(label, labelMap.size());
    }
    labelMap.put(defaultLabel, labelMap.size());
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
        correct = confusionMatrix[labelId][i];
      }
    }
    return 100.0 * correct / labelTotal;
  }
  
  public int getCorrect(String label) {
    int labelId = labelMap.get(label);
    return confusionMatrix[labelId][labelId];
  }
  
  public double getTotal(String label) {
    int labelId = labelMap.get(label);
    int labelTotal = 0;
    for (int i = 0; i < labelMap.size(); i++) {
      labelTotal += confusionMatrix[labelId][i];
    }
    return labelTotal;
  }
  
  public void addInstance(String correctLabel, ClassifierResult classifiedResult) {
    incrementCount(correctLabel, classifiedResult.getLabel());
  }
  
  public void addInstance(String correctLabel, String classifiedLabel) {
    incrementCount(correctLabel, classifiedLabel);
  }
  
  public int getCount(String correctLabel, String classifiedLabel) {
    Preconditions.checkArgument(labelMap.containsKey(correctLabel),
                                "Label not found: " + correctLabel);
    Preconditions.checkArgument(labelMap.containsKey(classifiedLabel),
                                "Label not found: " + classifiedLabel);
    int correctId = labelMap.get(correctLabel);
    int classifiedId = labelMap.get(classifiedLabel);
    return confusionMatrix[correctId][classifiedId];
  }
  
  public void putCount(String correctLabel, String classifiedLabel, int count) {
    Preconditions.checkArgument(labelMap.containsKey(correctLabel),
                                "Label not found: " + correctLabel);
    Preconditions.checkArgument(labelMap.containsKey(classifiedLabel),
                                "Label not found: " + classifiedLabel);
    int correctId = labelMap.get(correctLabel);
    int classifiedId = labelMap.get(classifiedLabel);
    confusionMatrix[correctId][classifiedId] = count;
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
      throw new CardinalityException(m.numRows(), m.numCols());
    }
    if (m.numRows() != length) {
      throw new CardinalityException(m.numRows(), length);
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
    labelMap.clear();    
	  if (labels != null) {
      labelMap.putAll(labels);
	  }
  }
  
  @Override
  public String toString() {
    StringBuilder returnString = new StringBuilder(200);
    returnString.append("=======================================================").append('\n');
    returnString.append("Confusion Matrix\n");
    returnString.append("-------------------------------------------------------").append('\n');
    
    for (Map.Entry<String,Integer> entry : this.labelMap.entrySet()) {
      returnString.append(StringUtils.rightPad(getSmallLabel(entry.getValue()), 5)).append('\t');
    }
    
    returnString.append("<--Classified as").append('\n');
    
    for (Map.Entry<String,Integer> entry : this.labelMap.entrySet()) {
      String correctLabel = entry.getKey();
      int labelTotal = 0;
      for (String classifiedLabel : this.labelMap.keySet()) {
        returnString.append(
          StringUtils.rightPad(Integer.toString(getCount(correctLabel, classifiedLabel)), 5)).append('\t');
        labelTotal += getCount(correctLabel, classifiedLabel);
      }
      returnString.append(" |  ").append(StringUtils.rightPad(String.valueOf(labelTotal), 6)).append('\t')
          .append(StringUtils.rightPad(getSmallLabel(entry.getValue()), 5))
          .append(" = ").append(correctLabel).append('\n');
    }
    returnString.append("Default Category: ").append(defaultLabel).append(": ").append(
      labelMap.get(defaultLabel)).append('\n');
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
