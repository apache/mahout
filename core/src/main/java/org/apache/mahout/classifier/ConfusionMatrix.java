package org.apache.mahout.classifier;

import java.util.*;

import org.apache.commons.lang.StringUtils;
import org.apache.mahout.common.Summarizable;
/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to You under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

/**
 * The ConfusionMatrix Class stores the result of Classification of a Test Dataset.
 */
public class ConfusionMatrix implements Summarizable {

  Collection<String> labels = new ArrayList<String>();

  Map<String, Integer> labelMap = new HashMap<String, Integer>();

  int[][] confusionMatrix = null;

  public int[][] getConfusionMatrix() {
    return confusionMatrix;
  }

  public Collection<String> getLabels() {
    return labels;
  }
  
  public float getAccuracy(String label){
    int labelId = labelMap.get(label).intValue();
    int labelTotal = 0;
    int correct = 0;    
    for(int i = 0 ;i < labels.size() ;i++){
      labelTotal+= confusionMatrix[labelId][i];
      if(i==labelId) correct = confusionMatrix[labelId][i];
    }
    return (float)100 * correct / labelTotal;
  }
  
  public int getCorrect(String label){
    int labelId = labelMap.get(label).intValue();
    int correct = 0;    
    correct = confusionMatrix[labelId][labelId];
  
    return correct;
  }
  
  public float getTotal(String label){
    int labelId = labelMap.get(label).intValue();
    int labelTotal = 0;
    for(int i = 0 ;i < labels.size() ;i++){
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
  
  public void addInstance(String correctLabel, ClassifierResult classifiedResult) throws Exception{
    incrementCount(correctLabel, classifiedResult.getLabel());
  }  
  
  public void addInstance(String correctLabel, String classifiedLabel) throws Exception{
    incrementCount(correctLabel, classifiedLabel);
  }
  
  public int getCount(String correctLabel, String classifiedLabel)
      throws Exception {
    if (this.getLabels().contains(correctLabel)
        && this.getLabels().contains(classifiedLabel) == false) {
      //System.err.println(correctLabel + " " + classifiedLabel);
      throw new Exception("Label not found " +correctLabel + " " +classifiedLabel );
    }
    int correctId = labelMap.get(correctLabel).intValue();
    int classifiedId = labelMap.get(classifiedLabel).intValue();
    return confusionMatrix[correctId][classifiedId];
  }

  public void putCount(String correctLabel, String classifiedLabel, int count)
      throws Exception {
    if (this.getLabels().contains(correctLabel)
        && this.getLabels().contains(classifiedLabel) == false) {
      throw new Exception("Label not found");
    }
    int correctId = labelMap.get(correctLabel).intValue();
    int classifiedId = labelMap.get(classifiedLabel).intValue();
    confusionMatrix[correctId][classifiedId] = count;
  }

  public void incrementCount(String correctLabel, String classifiedLabel,
      int count) throws Exception {
    putCount(correctLabel, classifiedLabel, count
        + getCount(correctLabel, classifiedLabel));
  }

  public void incrementCount(String correctLabel, String classifiedLabel)
      throws Exception {
    incrementCount(correctLabel, classifiedLabel, 1);
  }

  public ConfusionMatrix Merge(ConfusionMatrix b) throws Exception {
    if (this.getLabels().size() != b.getLabels().size())
      throw new Exception("The Labels do not Match");

    if (this.getLabels().containsAll(b.getLabels()))
      ;
    for (String correctLabel : this.labels) {
      for (String classifiedLabel : this.labels) {
        incrementCount(correctLabel, classifiedLabel, b.getCount(correctLabel,
            classifiedLabel));
      }
    }
    return this;
  }

  public String summarize() throws Exception {
    StringBuilder returnString = new StringBuilder();
    returnString
        .append("=======================================================\n");
    returnString.append("Confusion Matrix\n");
    returnString
        .append("-------------------------------------------------------\n");

    for (String correctLabel : this.labels) {
      returnString.append(StringUtils.rightPad(getSmallLabel(labelMap.get(
          correctLabel).intValue()), 5)
          + "\t");
    }

    returnString.append("<--Classified as\n");

    for (String correctLabel : this.labels) {
      Integer labelTotal = 0;
      for (String classifiedLabel : this.labels) {
        returnString.append(StringUtils.rightPad(new Integer(getCount(
            correctLabel, classifiedLabel)).toString(), 5)
            + "\t");
        labelTotal+=getCount(correctLabel, classifiedLabel);
      }
      returnString.append(" |  "
          + StringUtils.rightPad(labelTotal.toString(), 6)
          + "\t"
          + StringUtils.rightPad(getSmallLabel(labelMap.get(correctLabel)
              .intValue()), 5) + " = " + correctLabel + "\n");
    }
    returnString.append("\n");
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
