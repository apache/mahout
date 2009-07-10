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

import org.apache.commons.lang.StringUtils;
import org.apache.mahout.common.Summarizable;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * The ConfusionMatrix Class stores the result of Classification of a Test Dataset.
 *
 * See http://en.wikipedia.org/wiki/Confusion_matrix for background
 */
public class ConfusionMatrix implements Summarizable {

  private final Collection<String> labels;

  private final Map<String, Integer> labelMap = new HashMap<String, Integer>();

  private int[][] confusionMatrix = null;
  private String defaultLabel = "unknown";

  public ConfusionMatrix(Collection<String> labels, String defaultLabel) {
    this.labels = labels;
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
    return labels;
  }

  public double getAccuracy(String label) {
    int labelId = labelMap.get(label);
    int labelTotal = 0;
    int correct = 0;
    for (int i = 0; i < labels.size(); i++) {
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
    for (int i = 0; i < labels.size(); i++) {
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
    if (labels.contains(correctLabel)
        && labels.contains(classifiedLabel) == false && defaultLabel.equals(classifiedLabel) == false) {
      throw new IllegalArgumentException("Label not found " + correctLabel + ' ' + classifiedLabel);
    }
    int correctId = labelMap.get(correctLabel);
    int classifiedId = labelMap.get(classifiedLabel);
    return confusionMatrix[correctId][classifiedId];
  }

  public void putCount(String correctLabel, String classifiedLabel, int count) {
    if (labels.contains(correctLabel)
        && labels.contains(classifiedLabel) == false && defaultLabel.equals(classifiedLabel) == false) {
      throw new IllegalArgumentException("Label not found");
    }
    int correctId = labelMap.get(correctLabel);
    int classifiedId = labelMap.get(classifiedLabel);
    confusionMatrix[correctId][classifiedId] = count;
  }

  public void incrementCount(String correctLabel, String classifiedLabel,
                             int count) {
    putCount(correctLabel, classifiedLabel, count
        + getCount(correctLabel, classifiedLabel));
  }

  public void incrementCount(String correctLabel, String classifiedLabel) {
    incrementCount(correctLabel, classifiedLabel, 1);
  }

  public ConfusionMatrix merge(ConfusionMatrix b) {
    if (labels.size() != b.getLabels().size()) {
      throw new IllegalArgumentException("The Labels do not Match");
    }

    //if (labels.containsAll(b.getLabels()))
    //  ;
    for (String correctLabel : this.labels) {
      for (String classifiedLabel : this.labels) {
        incrementCount(correctLabel, classifiedLabel, b.getCount(correctLabel,
            classifiedLabel));
      }
    }
    return this;
  }

  @Override
  public String summarize() {
    String lineSep = System.getProperty("line.separator");
    StringBuilder returnString = new StringBuilder();
    returnString
        .append("=======================================================").append(lineSep);
    returnString.append("Confusion Matrix\n");
    returnString
        .append("-------------------------------------------------------").append(lineSep);

    for (String correctLabel : this.labels) {
      returnString.append(StringUtils.rightPad(getSmallLabel(labelMap.get(correctLabel)), 5)).append('\t');
    }

    returnString.append("<--Classified as").append(lineSep);

    for (String correctLabel : this.labels) {
      int labelTotal = 0;
      for (String classifiedLabel : this.labels) {
        returnString.append(StringUtils.rightPad(Integer.toString(getCount(
            correctLabel, classifiedLabel)), 5)).append('\t');
        labelTotal += getCount(correctLabel, classifiedLabel);
      }
      returnString.append(" |  ").append(StringUtils.rightPad(String.valueOf(labelTotal), 6)).append('\t')
          .append(StringUtils.rightPad(getSmallLabel(labelMap.get(correctLabel)), 5))
          .append(" = ").append(correctLabel).append(lineSep);
    }
    returnString.append("Default Category: ").append(defaultLabel).append(": ").append(labelMap.get(defaultLabel)).append(lineSep);
    returnString.append(lineSep);
    return returnString.toString();
  }

  static String getSmallLabel(int i) {
    int val = i;
    StringBuilder returnString = new StringBuilder();
    do {
      int n = val % 26;
      int c = 'a';
      returnString.insert(0, (char) (c + n));
      val /= 26;
    } while (val > 0);
    return returnString.toString();
  }

}
