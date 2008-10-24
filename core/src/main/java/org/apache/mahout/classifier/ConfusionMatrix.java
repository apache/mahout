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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * The ConfusionMatrix Class stores the result of Classification of a Test Dataset.
 */
public class ConfusionMatrix implements Summarizable {

  private Collection<String> labels = new ArrayList<String>();

  private final Map<String, Integer> labelMap = new HashMap<String, Integer>();

  private int[][] confusionMatrix = null;

  public int[][] getConfusionMatrix() {
    return confusionMatrix;
  }

  public Collection<String> getLabels() {
    return labels;
  }
  
  public double getAccuracy(String label){
    int labelId = labelMap.get(label);
    int labelTotal = 0;
    int correct = 0;    
    for(int i = 0 ;i < labels.size() ;i++){
      labelTotal += confusionMatrix[labelId][i];
      if(i == labelId)
        correct = confusionMatrix[labelId][i];
    }
    return 100.0 * correct / labelTotal;
  }
  
  public int getCorrect(String label){
    int labelId = labelMap.get(label);
    return confusionMatrix[labelId][labelId];
  }
  
  public double getTotal(String label){
    int labelId = labelMap.get(label);
    int labelTotal = 0;
    for (int i = 0 ;i < labels.size() ;i++){
      labelTotal+= confusionMatrix[labelId][i];
    }
    return labelTotal;
  }
  

  public ConfusionMatrix(Collection<String> labels) {
    this.labels = labels;
    confusionMatrix = new int[labels.size()][labels.size()];
    for (String label : labels) {
      labelMap.put(label, labelMap.size());
    }
  }
  
  public void addInstance(String correctLabel, ClassifierResult classifiedResult) {
    incrementCount(correctLabel, classifiedResult.getLabel());
  }  
  
  public void addInstance(String correctLabel, String classifiedLabel) {
    incrementCount(correctLabel, classifiedLabel);
  }
  
  public int getCount(String correctLabel, String classifiedLabel) {
    if (this.getLabels().contains(correctLabel)
        && this.getLabels().contains(classifiedLabel) == false) {
      throw new IllegalArgumentException("Label not found " +correctLabel + " " +classifiedLabel );
    }
    int correctId = labelMap.get(correctLabel);
    int classifiedId = labelMap.get(classifiedLabel);
    return confusionMatrix[correctId][classifiedId];
  }

  public void putCount(String correctLabel, String classifiedLabel, int count) {
    if (this.getLabels().contains(correctLabel)
        && this.getLabels().contains(classifiedLabel) == false) {
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

  public ConfusionMatrix Merge(ConfusionMatrix b) {
    if (this.getLabels().size() != b.getLabels().size())
      throw new IllegalArgumentException("The Labels do not Match");

    //if (this.getLabels().containsAll(b.getLabels()))
    //  ;
    for (String correctLabel : this.labels) {
      for (String classifiedLabel : this.labels) {
        incrementCount(correctLabel, classifiedLabel, b.getCount(correctLabel,
            classifiedLabel));
      }
    }
    return this;
  }

  public String summarize() {
    StringBuilder returnString = new StringBuilder();
    returnString
        .append("=======================================================\n");
    returnString.append("Confusion Matrix\n");
    returnString
        .append("-------------------------------------------------------\n");

    for (String correctLabel : this.labels) {
      returnString.append(StringUtils.rightPad(getSmallLabel(labelMap.get(correctLabel)), 5)).append('\t');
    }

    returnString.append("<--Classified as\n");

    for (String correctLabel : this.labels) {
      int labelTotal = 0;
      for (String classifiedLabel : this.labels) {
        returnString.append(StringUtils.rightPad(Integer.toString(getCount(
            correctLabel, classifiedLabel)), 5)).append('\t');
        labelTotal += getCount(correctLabel, classifiedLabel);
      }
      returnString.append(" |  ").append(StringUtils.rightPad(String.valueOf(labelTotal), 6)).append('\t')
          .append(StringUtils.rightPad(getSmallLabel(labelMap.get(correctLabel)), 5))
          .append(" = ").append(correctLabel).append('\n');
    }
    returnString.append('\n');
    return returnString.toString();
  }

  String getSmallLabel(int i) {
    int val = i;
    StringBuilder returnString = new StringBuilder();
    do{
      int n = val % 26;
      int c = 'a';
      returnString.insert(0, (char)(c + n));
      val /= 26;
    }while(val>0);
    return returnString.toString();
  }

}
