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
import java.util.List;
import java.util.PriorityQueue;

import org.apache.commons.lang.mutable.MutableDouble;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.common.ByScoreLabelResultComparator;
import org.apache.mahout.classifier.bayes.exceptions.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.interfaces.Algorithm;
import org.apache.mahout.classifier.bayes.interfaces.Datastore;
import org.apache.mahout.math.function.ObjectIntProcedure;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

/**
 * Class implementing the Naive Bayes Classifier Algorithm
 * 
 */
public class BayesAlgorithm implements Algorithm {
  
  @Override
  public ClassifierResult classifyDocument(String[] document,
                                           Datastore datastore,
                                           String defaultCategory) throws InvalidDatastoreException {
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
                                             Datastore datastore,
                                             String defaultCategory,
                                             int numResults) throws InvalidDatastoreException {
    Collection<String> categories = datastore.getKeys("labelWeight");
    PriorityQueue<ClassifierResult> pq = new PriorityQueue<ClassifierResult>(numResults,
        new ByScoreLabelResultComparator());
    for (String category : categories) {
      double prob = documentWeight(datastore, category, document);
      if (prob > 0.0) {
        pq.add(new ClassifierResult(category, prob));
        if (pq.size() > numResults) {
          pq.remove();
        }
      }
    }
    
    if (pq.isEmpty()) {
      return new ClassifierResult[] {new ClassifierResult(defaultCategory, 0.0)};
    } else {
      List<ClassifierResult> result = new ArrayList<ClassifierResult>(pq.size());
      while (!pq.isEmpty()) {
        result.add(pq.remove());
      }
      Collections.reverse(result);
      return result.toArray(new ClassifierResult[pq.size()]);
    }
  }
  
  @Override
  public double featureWeight(Datastore datastore, String label, String feature) throws InvalidDatastoreException {
    double result = datastore.getWeight("weight", feature, label);
    double vocabCount = datastore.getWeight("sumWeight", "vocabCount");
    double sumLabelWeight = datastore.getWeight("labelWeight", label);
    double numerator = result + datastore.getWeight("params", "alpha_i");
    double denominator = sumLabelWeight + vocabCount;
    double weight = Math.log(numerator / denominator);
    return -weight;
  }
  
  @Override
  public void initialize(Datastore datastore) throws InvalidDatastoreException {
    datastore.getKeys("labelWeight");
  }
  
  @Override
  public double documentWeight(final Datastore datastore,
                               final String label,
                               String[] document) {
    OpenObjectIntHashMap<String> wordList = new OpenObjectIntHashMap<String>(document.length / 2);
    for (String word : document) {
      if (wordList.containsKey(word)) {
        wordList.put(word, wordList.get(word) + 1);
      } else {
        wordList.put(word, 1);
      }
    }
    final MutableDouble result = new MutableDouble(0.0);
    
    wordList.forEachPair(new ObjectIntProcedure<String>() {
      
      @Override
      public boolean apply(String word, int frequency) {
        try {
          result.add(frequency * featureWeight(datastore, label, word));
        } catch (InvalidDatastoreException e) {
          throw new IllegalStateException(e);
        }
        return true;
      }
    });
    return result.doubleValue();
  }
  
  @Override
  public Collection<String> getLabels(Datastore datastore) throws InvalidDatastoreException {
    return datastore.getKeys("labelWeight");
  }
  
}
