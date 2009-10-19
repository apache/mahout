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

package org.apache.mahout.classifier.bayes.algorithm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.common.ByScoreLabelResultComparator;
import org.apache.mahout.classifier.bayes.exceptions.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.interfaces.Algorithm;
import org.apache.mahout.classifier.bayes.interfaces.Datastore;

public class BayesAlgorithm implements Algorithm{

  @Override
  public ClassifierResult classifyDocument(String[] document,
      Datastore datastore, String defaultCategory)
      throws InvalidDatastoreException {
    ClassifierResult result = new ClassifierResult(defaultCategory);
    double max = Double.MAX_VALUE;
    Collection<String> categories = datastore.getKeys("labelWeight");    

    for (String category : categories) {
      double prob = documentWeight(datastore, category, document);
      if (prob < max) {
        max = prob;
        result.setLabel(category);
      }
    }
    result.setScore(max);
    return result;
  }

  @Override
  public ClassifierResult[] classifyDocument(String[] document,
      Datastore datastore, String defaultCategory, int numResults)
      throws InvalidDatastoreException {
    Collection<String> categories = datastore.getKeys("labelWeight");    
    PriorityQueue<ClassifierResult> pq =
        new PriorityQueue<ClassifierResult>(numResults, new ByScoreLabelResultComparator());
    for (String category : categories){
      double prob = documentWeight(datastore, category, document);
      if (prob > 0.0) {
        pq.add(new ClassifierResult(category, prob));
        if (pq.size() > numResults) {
          pq.remove();
        }
      }
    }

    if (pq.isEmpty()) {
      return new ClassifierResult[] { new ClassifierResult(defaultCategory, 0.0) };
    } else {
      List<ClassifierResult> result = new ArrayList<ClassifierResult>(pq.size());
      while (pq.isEmpty() == false) {
        result.add(pq.remove());
      }
      Collections.reverse(result);
      return result.toArray(new ClassifierResult[pq.size()]);
    }
  }
  
  @Override
  public double featureWeight(Datastore datastore, String label, String feature)
      throws InvalidDatastoreException {       
    
    double result = datastore.getWeight("weight", feature, label);
    double vocabCount = datastore.getWeight("sumWeight", "vocabCount");
    double sumLabelWeight = datastore.getWeight("labelWeight", label);    
    double numerator = result + datastore.getWeight("params", "alpha_i");
    double denominator = (sumLabelWeight + vocabCount);
    double weight = Math.log(numerator / denominator);
    result = -weight;
    return result;
  }

  @Override
  public void initialize(Datastore datastore) throws InvalidDatastoreException {
    datastore.getWeight("weight", "test", "test");
    datastore.getWeight("labelWeight", "test");
    datastore.getWeight("thetaNormalizer", "test");
  }

  @Override
  public double documentWeight(Datastore datastore, String label,
      String[] document) throws InvalidDatastoreException {
    Map<String, int[]> wordList = new HashMap<String, int[]>(1000);
    for (String word : document) {
      int[] count = wordList.get(word);
      if (count == null) {
        count = new int[] { 0 };
        wordList.put(word, count);
      }
      count[0]++;
    }
    double result = 0.0;
    for (Map.Entry<String, int[]> entry : wordList.entrySet()) {
      String word = entry.getKey();
      int count = entry.getValue()[0];
      result += count * featureWeight(datastore, label, word);
    }
    return result;
  }

  @Override
  public Collection<String> getLabels(Datastore datastore)
      throws InvalidDatastoreException {
    return datastore.getKeys("labelWeight");
  }


}
